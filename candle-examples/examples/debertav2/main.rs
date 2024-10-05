#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::fmt::Display;
use std::path::PathBuf;

use anyhow::ensure;
use anyhow::{Error as E, Result};
use candle::{Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Id2Label;
use candle_transformers::models::debertav2::NERItem;
use candle_transformers::models::debertav2::{Config as DebertaV2Config, DebertaV2NERModel};
use clap::{ArgGroup, Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{Encoding, PaddingParams, Tokenizer};

enum TaskType {
    NER(DebertaV2NERModel),
}

#[derive(Parser, Debug, Clone, ValueEnum)]
enum ArgsTask {
    /// Named Entity Recognition
    NER,
}

impl Display for ArgsTask {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ArgsTask::NER => write!(f, "ner"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(group(ArgGroup::new("model")
    .required(true)
    .args(&["model_id", "model_path"])))]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model id to use from HuggingFace
    #[arg(long, requires_if("model_id", "revision"))]
    model_id: Option<String>,

    /// Revision of the model to use (default: "main")
    #[arg(long, default_value = "main")]
    revision: String,

    /// Specify a sentence to inference. Specify multiple times to inference multiple sentences.
    #[arg(long = "sentence", name="sentences", num_args = 1..)]
    sentences: Vec<String>,

    /// Use the pytorch weights rather than the by-default safetensors
    #[arg(long)]
    use_pth: bool,

    /// Perform a very basic benchmark on inferencing, using N number of iterations
    #[arg(long)]
    benchmark_iters: Option<usize>,

    // /// Use tanh based approximation for Gelu instead of erf implementation.
    // #[arg(long, default_value = "false")]
    // approximate_gelu: bool,
    /// Which task to run
    #[arg(long, default_value_t = ArgsTask::NER)]
    task: ArgsTask,

    /// Use model from a specific directory instead of HuggingFace local cache.
    /// Using this ignores model_id and revision args.
    #[arg(long)]
    model_path: Option<PathBuf>,
}

impl Args {
    fn build_model_and_tokenizer(
        &self,
        id2label: Option<Id2Label>,
    ) -> Result<(TaskType, DebertaV2Config, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;

        // Get files from either the HuggingFace API, or from a specified local directory.
        let (config_filename, tokenizer_filename, weights_filename) = {
            match &self.model_path {
                Some(base_path) => {
                    ensure!(
                        base_path.is_dir(),
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Model path {} is not a directory.", base_path.display()),
                        )
                    );

                    let config = base_path.join("config.json");
                    let tokenizer = base_path.join("tokenizer.json");
                    let weights = if self.use_pth {
                        base_path.join("pytorch_model.bin")
                    } else {
                        base_path.join("model.safetensors")
                    };
                    (config, tokenizer, weights)
                }
                None => {
                    let repo = Repo::with_revision(
                        self.model_id.as_ref().unwrap().clone(),
                        RepoType::Model,
                        self.revision.clone(),
                    );
                    let api = Api::new()?;
                    let api = api.repo(repo);
                    let config = api.get("config.json")?;
                    let tokenizer = api.get("tokenizer.json")?;
                    let weights = if self.use_pth {
                        api.get("pytorch_model.bin")?
                    } else {
                        api.get("model.safetensors")?
                    };
                    (config, tokenizer, weights)
                }
            }
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: DebertaV2Config = serde_json::from_str(&config)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle::Error::Msg(format!("Tokenizer error: {e}")))?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        let vb = if self.use_pth {
            VarBuilder::from_pth(
                &weights_filename,
                candle_transformers::models::debertav2::DTYPE,
                &device,
            )?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_filename],
                    candle_transformers::models::debertav2::DTYPE,
                    &device,
                )?
            }
        };

        let vb = vb.set_prefix("deberta");

        // TODO: Should we?
        // if self.approximate_gelu {
        //     config.hidden_act = HiddenAct::GeluApproximate;
        // }

        match self.task {
            ArgsTask::NER => Ok((
                TaskType::NER(DebertaV2NERModel::load(vb, &config, id2label)?),
                config,
                tokenizer,
            )),
        }
    }
}

fn get_device(model_type: &TaskType) -> &Device {
    match model_type {
        TaskType::NER(ner_model) => &ner_model.device,
    }
}

enum InputEncoding {
    Single(Encoding),
    Batch(Vec<Encoding>),
}

struct ModelInput {
    encoding: InputEncoding,
    input_ids: Tensor,
    attention_mask: Tensor,
    token_type_ids: Tensor,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    if args.model_id.is_some() && args.model_path.is_some() {
        eprintln!("Error: Cannot specify both --model_id and --model_path.");
        std::process::exit(1);
    }

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let model_load_time = std::time::Instant::now();
    let (task_type, model_config, tokenizer) = args.build_model_and_tokenizer(None)?;
    println!(
        "Loaded model and tokenizers in {:?}",
        model_load_time.elapsed()
    );

    let device = get_device(&task_type);

    // Single sentence passed in means we don't need batching.
    // Multiple sentences passed in means we have can/should do batching.
    let tokenize_time = std::time::Instant::now();
    let model_input: ModelInput = match args.sentences.len() {
        1 => {
            let encoding = tokenizer
                .encode(args.sentences.first().unwrap().as_str(), true)
                .map_err(E::msg)?;

            ModelInput {
                input_ids: Tensor::new(&encoding.get_ids()[..], &device)?.unsqueeze(0)?,
                attention_mask: Tensor::new(&encoding.get_attention_mask()[..], &device)?
                    .unsqueeze(0)?,
                token_type_ids: Tensor::new(&encoding.get_type_ids()[..], &device)?.unsqueeze(0)?,
                encoding: InputEncoding::Single(encoding),
            }
        }
        _ => {
            let tokenizer_encodings = tokenizer
                .encode_batch(args.sentences, true)
                .map_err(E::msg)?;

            let mut encoding_stack: Vec<Tensor> = Vec::default();
            let mut attention_mask_stack: Vec<Tensor> = Vec::default();
            let mut token_type_id_stack: Vec<Tensor> = Vec::default();

            for encoding in &tokenizer_encodings {
                encoding_stack.push(Tensor::new(encoding.get_ids(), &device)?);
                attention_mask_stack.push(Tensor::new(encoding.get_attention_mask(), &device)?);
                token_type_id_stack.push(Tensor::new(encoding.get_type_ids(), &device)?);
            }

            ModelInput {
                encoding: InputEncoding::Batch(tokenizer_encodings),
                input_ids: Tensor::stack(&encoding_stack[..], 0)?,
                attention_mask: Tensor::stack(&attention_mask_stack[..], 0)?,
                token_type_ids: Tensor::stack(&token_type_id_stack[..], 0)?,
            }
        }
    };

    println!(
        "Tokenized and loaded inputs in {:?}",
        tokenize_time.elapsed()
    );

    match task_type {
        TaskType::NER(ner_model) => {
            if let Some(num_iters) = args.benchmark_iters {
                create_benchmark(num_iters, model_input)(
                    |input_ids, token_type_ids, attention_mask| {
                        ner_model
                            .forward(input_ids, Some(token_type_ids), Some(attention_mask))
                            .expect("ohno");
                        Ok(())
                    },
                );

                std::process::exit(0);
            }

            let inference_time = std::time::Instant::now();
            let logits = ner_model.forward(
                &model_input.input_ids,
                Some(model_input.token_type_ids),
                Some(model_input.attention_mask),
            )?;
            println!("Inferenced inputs in {:?}", inference_time.elapsed());

            match model_input.encoding {
                InputEncoding::Single(tokenizer_encoding) => {
                    let maxes = logits.max_keepdim(D::Minus1)?;
                    let shifted_exp = {
                        let logits_minus_maxes = logits.broadcast_sub(&maxes)?;
                        logits_minus_maxes.exp()?
                    };
                    let scores = {
                        let sum = shifted_exp.sum_keepdim(D::Minus1)?;
                        shifted_exp.broadcast_div(&sum)?.squeeze(0)?
                    };

                    let predicted_label_ids = scores.argmax(1)?.to_vec1::<u32>()?;
                    let max_scores = scores.max(1)?.to_vec1::<f32>()?;
                    let id2label = model_config.id2label.as_ref().unwrap();
                    let mut result: Vec<NERItem> = Vec::default();

                    for (idx, predicted_label_id) in predicted_label_ids.iter().enumerate() {
                        if tokenizer_encoding.get_special_tokens_mask()[idx] == 1 {
                            continue;
                        }

                        let label = id2label.get(&predicted_label_id).unwrap();

                        if label == "O" {
                            continue;
                        }

                        result.push(NERItem {
                            entity: label.clone(),
                            word: tokenizer_encoding.get_tokens()[idx].clone(),
                            score: max_scores[idx],
                            start: tokenizer_encoding.get_offsets()[idx].0,
                            end: tokenizer_encoding.get_offsets()[idx].1,
                            index: idx,
                        });
                    }
                    println!("\n{:?}", result);
                }
                InputEncoding::Batch(tokenizer_encodings) => {
                    let maxes = logits.max_keepdim(D::Minus1)?;
                    let shifted_exp = {
                        let logits_minus_maxes = logits.broadcast_sub(&maxes)?;
                        logits_minus_maxes.exp()?
                    };
                    let scores = {
                        let sum = shifted_exp.sum_keepdim(D::Minus1)?;
                        shifted_exp.broadcast_div(&sum)?
                    };

                    let max_scores = scores.max(D::Minus1)?.to_vec2::<f32>()?;

                    let batch_scores = scores.argmax_keepdim(2)?.to_vec3::<u32>()?;

                    let mut batch_results: Vec<Vec<NERItem>> = Vec::default();

                    for (batch_idx, batch_score) in batch_scores.iter().enumerate() {
                        let mut batch_result: Vec<NERItem> = Vec::default();
                        for (idx, input_label) in batch_score.iter().enumerate() {
                            if tokenizer_encodings[batch_idx].get_special_tokens_mask()[idx] == 1 {
                                continue;
                            }

                            let label =
                                model_config.id2label.as_ref().unwrap()[&input_label[0]].clone();

                            if label == "O" {
                                continue;
                            }

                            batch_result.push(NERItem {
                                entity: label,
                                word: tokenizer_encodings[batch_idx].get_tokens()[idx].clone(),
                                score: max_scores[batch_idx][idx],
                                start: tokenizer_encodings[batch_idx].get_offsets()[idx].0,
                                end: tokenizer_encodings[batch_idx].get_offsets()[idx].1,
                                index: idx,
                            })
                        }
                        batch_results.push(batch_result);
                    }

                    println!("\n{:?}", batch_results);
                }
            }
        }
    }
    Ok(())
}

fn create_benchmark<F>(num_iters: usize, model_input: ModelInput) -> impl Fn(F) -> ()
where
    F: Fn(&Tensor, Tensor, Tensor) -> Result<(), candle::Error>,
{
    move |code: F| -> () {
        println!("Running {num_iters} iterations...");
        let mut durations = Vec::with_capacity(num_iters);
        for _ in 0..num_iters {
            let token_type_ids = model_input.token_type_ids.clone();
            let attention_mask = model_input.attention_mask.clone();
            let start = std::time::Instant::now();
            code(&model_input.input_ids, token_type_ids, attention_mask).expect("ohno");
            let duration = start.elapsed();
            durations.push(duration.as_nanos());
        }

        let min_time = *durations.iter().min().unwrap();
        let max_time = *durations.iter().max().unwrap();
        let avg_time = durations.iter().sum::<u128>() as f64 / num_iters as f64;

        println!("Min time: {:.3} ms", min_time as f64 / 1_000_000.0);
        println!("Avg time: {:.3} ms", avg_time / 1_000_000.0);
        println!("Max time: {:.3} ms", max_time as f64 / 1_000_000.0);
    }
}

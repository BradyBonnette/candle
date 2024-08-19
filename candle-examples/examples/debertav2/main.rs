#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::collections::HashMap;
use std::fmt::Display;
use std::path::PathBuf;

use anyhow::ensure;
use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{
    Config as DebertaV2Config, DebertaV2NERModel, SentencePiece,
};
use candle_transformers::models::debertav2::{DebertaV2Model, Id2Label};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use sentencepiece::SentencePieceProcessor;
use tokenizers::PaddingParams;

enum DebertaV2ModelType {
    Base(DebertaV2Model),
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
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use
    #[arg(long, default_value = "microsoft/deberta-v3-large")]
    model_id: String,

    /// Revision of the model to use
    #[arg(long, default_value = "refs/pr/4")]
    revision: String,

    /// Specify a sentence to inference. Specify multiple times to inference multiple sentences.
    #[arg(long = "sentence", name="sentences", num_args = 1..)]
    sentences: Vec<String>,

    /// Use the pytorch weights rather than the by-default safetensors
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    // /// Use tanh based approximation for Gelu instead of erf implementation.
    // #[arg(long, default_value = "false")]
    // approximate_gelu: bool,
    /// Which task to run
    #[arg(long, default_value_t = ArgsTask::NER)]
    task: ArgsTask,

    /// Use model from specific directory instead of HuggingFace local cache.
    /// Using this ignores model_id and revision args.
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Specify SentencePiece model filename
    #[arg(long, default_value = "spm.model")]
    spm_filename: String,
}

impl Args {
    fn build_model_and_tokenizer(
        &self,
        id2label: Option<Id2Label>,
    ) -> Result<(DebertaV2ModelType, DebertaV2Config, SentencePieceProcessor)> {
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
                    let tokenizer = base_path.join(self.spm_filename.as_str());
                    let weights = if self.use_pth {
                        base_path.join("pytorch_model.bin")
                    } else {
                        base_path.join("model.safetensors")
                    };
                    (config, tokenizer, weights)
                }
                None => {
                    let repo = Repo::with_revision(
                        self.model_id.clone(),
                        RepoType::Model,
                        self.revision.clone(),
                    );
                    let api = Api::new()?;
                    let api = api.repo(repo);
                    let config = api.get("config.json")?;
                    let tokenizer = api.get(self.spm_filename.as_str())?;
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

        let tokenizer = SentencePieceProcessor::open(tokenizer_filename)?;

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
                DebertaV2ModelType::NER(DebertaV2NERModel::load(vb, &config, id2label)?),
                config,
                tokenizer,
            )),
        }
    }
}

fn get_device(model_type: &DebertaV2ModelType) -> &Device {
    match model_type {
        DebertaV2ModelType::Base(base_model) => &base_model.device,
        DebertaV2ModelType::NER(ner_model) => &ner_model.device,
    }
}

fn special_tokens(spp: &SentencePieceProcessor) -> HashMap<u32, bool> {
    let mut special_tokens = HashMap::<u32, bool>::new();
    if let Some(id) = spp.bos_id() {
        special_tokens.insert(id, true);
    }
    if let Some(id) = spp.eos_id() {
        special_tokens.insert(id, true);
    }
    if let Some(id) = spp.pad_id() {
        special_tokens.insert(id, true);
    }
    special_tokens.insert(spp.unk_id(), true);

    special_tokens
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    println!("{:?}", args.sentences);

    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // let start = std::time::Instant::now();

    let (model_type, model_config, tokenizer) = args.build_model_and_tokenizer(None)?;
    // let device = get_device(&model_type);

    let mut token_pieces: Vec<candle_transformers::models::debertav2::SentencePiece> = tokenizer
        .encode(args.sentences.first().unwrap())?
        .iter()
        .map(|piece| SentencePiece {
            piece: piece.piece.clone(),
            id: piece.id,
            span: piece.span,
            is_special: false,
        })
        .collect();

    token_pieces.insert(
        0,
        SentencePiece {
            piece: "".to_string(),
            id: tokenizer.bos_id().unwrap(),
            span: (0, 0),
            is_special: true,
        },
    );

    token_pieces.push(SentencePiece {
        piece: String::new(),
        id: tokenizer.eos_id().unwrap(),
        span: (0, 0),
        is_special: true,
    });

    assert!(
        token_pieces.len() - 2 <= model_config.max_position_embeddings,
        "Number of tokens produced for sentence ```{}``` ({} tokens) is larger than the model's configuration for max embeddings ({} tokens).",
        args.sentences.first().unwrap(),
        token_pieces.len(),
        model_config.max_position_embeddings
    );

    // let mut token_ids: Vec<u32> = vec![tokenizer.bos_id().unwrap()];
    // token_pieces
    //     .iter()
    //     .for_each(|piece| token_ids.push(piece.id));
    // token_ids.push(tokenizer.eos_id().unwrap());

    // let input: Tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

    match model_type {
        DebertaV2ModelType::Base(_base_model) => todo!(),
        DebertaV2ModelType::NER(ner_model) => {
            let entities = ner_model.entities(&token_pieces, None, None)?;
            println!("{:?}", entities);
        }
    }

    // let device = match model {
    //     DebertaV2ModelType::Base(base_model) => base_model.device,
    //     DebertaV2ModelType::NER(ner_model) => ner_model.device,
    // };

    // if let Some(prompt) = args.prompt {
    //     let tokenizer = tokenizer
    //         .with_padding(None)
    //         .with_truncation(None)
    //         .map_err(E::msg)?;
    //     let tokens = tokenizer
    //         .encode(prompt, true)
    //         .map_err(E::msg)?
    //         .get_ids()
    //         .to_vec();
    //     let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    //     let token_type_ids = token_ids.zeros_like()?;
    //     println!("Loaded and encoded {:?}", start.elapsed());
    //     for idx in 0..args.n {
    //         let start = std::time::Instant::now();
    //         let ys = model.forward(&token_ids, &token_type_ids, None)?;
    //         if idx == 0 {
    //             println!("{ys}");
    //         }
    //         println!("Took {:?}", start.elapsed());
    //     }
    // } else {
    //     let sentences = [
    //         "The cat sits outside",
    //         "A man is playing guitar",
    //         "I love pasta",
    //         "The new movie is awesome",
    //         "The cat plays in the garden",
    //         "A woman watches TV",
    //         "The new movie is so great",
    //         "Do you like pizza?",
    //     ];
    //     let n_sentences = sentences.len();
    //     if let Some(pp) = tokenizer.get_padding_mut() {
    //         pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    //     } else {
    //         let pp = PaddingParams {
    //             strategy: tokenizers::PaddingStrategy::BatchLongest,
    //             ..Default::default()
    //         };
    //         tokenizer.with_padding(Some(pp));
    //     }
    //     let tokens = tokenizer
    //         .encode_batch(sentences.to_vec(), true)
    //         .map_err(E::msg)?;
    //     let token_ids = tokens
    //         .iter()
    //         .map(|tokens| {
    //             let tokens = tokens.get_ids().to_vec();
    //             Ok(Tensor::new(tokens.as_slice(), device)?)
    //         })
    //         .collect::<Result<Vec<_>>>()?;
    //     let attention_mask = tokens
    //         .iter()
    //         .map(|tokens| {
    //             let tokens = tokens.get_attention_mask().to_vec();
    //             Ok(Tensor::new(tokens.as_slice(), device)?)
    //         })
    //         .collect::<Result<Vec<_>>>()?;

    //     let token_ids = Tensor::stack(&token_ids, 0)?;
    //     let attention_mask = Tensor::stack(&attention_mask, 0)?;
    //     let token_type_ids = token_ids.zeros_like()?;
    //     println!("running inference on batch {:?}", token_ids.shape());
    //     let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
    //     println!("generated embeddings {:?}", embeddings.shape());
    //     // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    //     let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    //     let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    //     let embeddings = if args.normalize_embeddings {
    //         normalize_l2(&embeddings)?
    //     } else {
    //         embeddings
    //     };
    //     println!("pooled embeddings {:?}", embeddings.shape());

    //     let mut similarities = vec![];
    //     for i in 0..n_sentences {
    //         let e_i = embeddings.get(i)?;
    //         for j in (i + 1)..n_sentences {
    //             let e_j = embeddings.get(j)?;
    //             let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
    //             let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
    //             let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
    //             let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
    //             similarities.push((cosine_similarity, i, j))
    //         }
    //     }
    //     similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    //     for &(score, i, j) in similarities[..5].iter() {
    //         println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    //     }
    // }
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

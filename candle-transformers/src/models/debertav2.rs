// TEMPORARY
macro_rules! t {
    ($name:expr, $tensor:expr) => {
        // println!("\n{}:\n{}", $name, $tensor.to_string());
    };
}
// TEMPORARY

use std::collections::HashMap;

use candle::{DType, Device, Module, Tensor, D};
use candle_nn::{
    conv1d, embedding, layer_norm, Conv1d, Conv1dConfig, Embedding, LayerNorm, VarBuilder,
};
use serde::{Deserialize, Deserializer};

pub const DTYPE: DType = DType::F32;

// NOTE: HiddenAct and HiddenActLayer are both direct copies from bert.rs.
// If there's not much difference we could probably re-use them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

pub struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

// Arguments:
//     vocab_size (`int`, *optional*, defaults to 128100):
//         Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
//         the `inputs_ids` passed when calling [`DebertaV2Model`].
//     hidden_size (`int`, *optional*, defaults to 1536):
//         Dimensionality of the encoder layers and the pooler layer.
//     num_hidden_layers (`int`, *optional*, defaults to 24):
//         Number of hidden layers in the Transformer encoder.
//     num_attention_heads (`int`, *optional*, defaults to 24):
//         Number of attention heads for each attention layer in the Transformer encoder.
//     intermediate_size (`int`, *optional*, defaults to 6144):
//         Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
//     hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
//         The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
//         `"relu"`, `"silu"`, `"gelu"`, `"tanh"`, `"gelu_fast"`, `"mish"`, `"linear"`, `"sigmoid"` and `"gelu_new"`
//         are supported.
//     hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
//         The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
//     attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
//         The dropout ratio for the attention probabilities.
//     max_position_embeddings (`int`, *optional*, defaults to 512):
//         The maximum sequence length that this model might ever be used with. Typically set this to something large
//         just in case (e.g., 512 or 1024 or 2048).
//     type_vocab_size (`int`, *optional*, defaults to 0):
//         The vocabulary size of the `token_type_ids` passed when calling [`DebertaModel`] or [`TFDebertaModel`].
//     initializer_range (`float`, *optional*, defaults to 0.02):
//         The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
//     layer_norm_eps (`float`, *optional*, defaults to 1e-7):
//         The epsilon used by the layer normalization layers.
//     relative_attention (`bool`, *optional*, defaults to `True`):
//         Whether use relative position encoding.
//     max_relative_positions (`int`, *optional*, defaults to -1):
//         The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
//         as `max_position_embeddings`.
//     pad_token_id (`int`, *optional*, defaults to 0):
//         The value used to pad input_ids.
//     position_biased_input (`bool`, *optional*, defaults to `True`):
//         Whether add absolute position embedding to content embedding.
//     pos_att_type (`List[str]`, *optional*):
//         The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
//         `["p2c", "c2p"]`, `["p2c", "c2p"]`.
//     layer_norm_eps (`float`, optional, defaults to 1e-12):
//         The epsilon used by the layer normalization layers.

pub type Id2Label = HashMap<usize, String>;
pub type Label2Id = HashMap<String, usize>;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,          // (`int`, *optional*, defaults to 128100):
    pub hidden_size: usize,         //(`int`, *optional*, defaults to 1536):
    pub num_hidden_layers: usize,   // (`int`, *optional*, defaults to 24):
    pub num_attention_heads: usize, //(`int`, *optional*, defaults to 24):
    pub intermediate_size: usize,   //(`int`, *optional*, defaults to 6144):
    pub hidden_act: HiddenAct,      //(`str` or `Callable`, *optional*, defaults to `"gelu"`):
    pub hidden_dropout_prob: f64,   // (`float`, *optional*, defaults to 0.1):
    pub attention_probs_dropout_prob: f64, // (`float`, *optional*, defaults to 0.1):
    pub max_position_embeddings: usize, //(`int`, *optional*, defaults to 512):
    pub type_vocab_size: usize,     // (`int`, *optional*, defaults to 0):
    pub initializer_range: f64,     // (`float`, *optional*, defaults to 0.02):
    pub layer_norm_eps: f64,        // (`float`, *optional*, defaults to 1e-7):
    pub relative_attention: bool,   //(`bool`, *optional*, defaults to `True`):
    pub max_relative_positions: isize, //(`int`, *optional*, defaults to -1):
    pub pad_token_id: Option<usize>, //(`int`, *optional*, defaults to 0):
    pub position_biased_input: bool, // (`bool`, *optional*, defaults to `True`):
    #[serde(deserialize_with = "deserialize_pos_att_type")]
    pub pos_att_type: Vec<String>,
    pub position_buckets: Option<isize>,
    pub share_att_key: Option<bool>,
    pub attention_head_size: Option<usize>,
    pub embedding_size: Option<usize>,
    pub norm_rel_ebd: Option<String>,
    pub conv_kernel_size: Option<usize>,
    pub conv_groups: Option<usize>,
    pub conv_act: Option<String>,
    pub id2label: Option<Id2Label>,
    pub label2id: Option<Label2Id>,
}

fn deserialize_pos_att_type<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    // Define an intermediate enum to represent the possible input types
    #[derive(Deserialize, Debug)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<String>),
    }

    // Deserialize the input into the intermediate enum
    let parsed: StringOrVec = StringOrVec::deserialize(deserializer)?;

    // println!("parsed: {:?}", parsed);
    // Match on the enum to handle both cases
    match parsed {
        StringOrVec::String(s) => Ok(s.split('|').map(String::from).collect()),
        StringOrVec::Vec(v) => Ok(v),
    }
    // let s: String = String::deserialize(deserializer)?;
    // Ok(s.split('|').map(String::from).collect())
}

// TODO: Dropout is probably not needed for now since this will primarily be used
// in inferencing. However, for training/fine-tuning it will be necessary.
pub struct StableDropout {
    drop_prob: f64,
    count: usize,
}

impl StableDropout {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            drop_prob,
            count: 0,
        }
    }

    // pub fn forward(&self, x: Tensor) -> candle::Result<Tensor> {
    pub fn forward(&self, x: Option<&Tensor>) -> candle::Result<Option<Tensor>> {
        Ok(x.cloned())

        // Ok(x)
        // pub fn forward(&self, x: Option<&Tensor>) -> candle::Result<Tensor> {
        // Ok(x.cloned())
    }
}

pub struct DebertaV2Embeddings {
    device: Device,
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    dropout: StableDropout,
    position_ids: Tensor,
    config: Config,
    embedding_size: usize,
    embed_proj: Option<candle_nn::Linear>,
}

impl DebertaV2Embeddings {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let device = vb.device().clone();
        let config = config.clone();

        let embedding_size = match config.embedding_size {
            Some(es) => es,
            None => config.hidden_size,
        };

        let word_embeddings =
            embedding(config.vocab_size, embedding_size, vb.pp("word_embeddings"))?;

        let position_embeddings = match config.position_biased_input {
            true => Some(embedding(
                config.max_position_embeddings,
                embedding_size,
                vb.pp("position_embeddings"),
            )?),
            false => None,
        };

        let token_type_embeddings: Option<Embedding> = match config.type_vocab_size > 0 {
            true => Some(candle_nn::embedding(
                config.type_vocab_size,
                config.hidden_size,
                vb.pp("token_type_embeddings"),
            )?),
            false => None,
        };

        let embed_proj: Option<candle_nn::Linear> = match embedding_size != config.hidden_size {
            true => Some(candle_nn::linear_no_bias(
                embedding_size,
                config.hidden_size,
                vb.pp("embed_proj"),
            )?),
            false => None,
        };

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = StableDropout::new(config.hidden_dropout_prob);

        // TEMP: Verified
        let position_ids =
            Tensor::arange(0, config.max_position_embeddings as u32, &device)?.unsqueeze(0)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            position_ids,
            device,
            config,
            embedding_size,
            embed_proj,
        })
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
    ) -> candle::Result<Tensor> {
        let input_shape = match (input_ids, inputs_embeds) {
            (Some(inputids), None) => inputids.dims(),
            (None, Some(inputsembeds)) => inputsembeds.dims(),
            (None, None) => {
                return Err(candle::Error::Msg(
                    "Must specify either input_ids or inputs_embeds".to_string(),
                ))
            }
            (Some(_), Some(_)) => {
                return Err(candle::Error::Msg(
                    "Can't specify both input_ids and inputs_embeds".to_string(),
                ))
            }
        };

        // TEMP: Verified
        let seq_length = input_shape.last().unwrap().to_owned();

        t!("self.position_ids", self.position_ids);

        // TEMP: Verified
        let position_ids = match position_ids {
            Some(p) => p.to_owned(),
            // None => self.position_ids.narrow(0, 0, seq_length)?,
            None => self.position_ids.narrow(1, 0, seq_length)?,
        };

        // println!("position_ids dims: {:?}", position_ids.dims());
        // println!("position_ids: {}", position_ids.to_string());

        // TEMP: Verified
        let token_type_ids = match token_type_ids {
            Some(t) => t.to_owned(),
            None => Tensor::zeros(input_shape, DType::U32, &self.device)?,
        };

        // println!("token_type_ids dims: {:?}", token_type_ids.dims());
        // println!("token_type_ids: {}", token_type_ids.to_string());

        // TEMP: Verified
        let input_embeds = match inputs_embeds {
            Some(e) => e.to_owned(),
            None => self.word_embeddings.forward(input_ids.unwrap())?,
        };

        // println!("input_embeds dims: {:?}", input_embeds.dims());
        // println!("input_embeds: {}", input_embeds.to_string());

        // TEMP: Verified
        let position_embeddings = match &self.position_embeddings {
            Some(emb) => emb.forward(&position_ids)?,
            None => Tensor::zeros_like(&input_embeds)?,
        };

        let mut embeddings = input_embeds;

        // TEMP: Verified, but skipped
        if self.config.position_biased_input {
            embeddings = embeddings.add(&position_embeddings)?;
        }

        // TEMP: Verified, but skipped
        if self.config.type_vocab_size > 0 {
            let token_type_embeddings = self.token_type_embeddings.as_ref().unwrap();
            let token_type_embeddings = token_type_embeddings.forward(&token_type_ids)?;
            embeddings = embeddings.add(&token_type_embeddings)?;
        }

        // TEMP: Verfieid, but skipped
        if self.embedding_size != self.config.hidden_size {
            embeddings = self.embed_proj.as_ref().unwrap().forward(&embeddings)?;
        }

        // TEMP: Verified
        embeddings = self.layer_norm.forward(&embeddings)?;

        // Temp: Verified
        if let Some(mask) = mask {
            let mut mask = mask.clone();
            if mask.dims() != embeddings.dims() {
                if mask.dims().len() == 4 {
                    mask = mask.squeeze(1)?.squeeze(1)?;
                }
                mask = mask.unsqueeze(2)?;
            }

            mask = mask.to_dtype(embeddings.dtype())?;
            embeddings = embeddings.broadcast_mul(&mask)?;
        }

        embeddings = self.dropout.forward(Some(&embeddings))?.unwrap();

        Ok(embeddings)
    }
}

/*
fn masked_fill(input: &Tensor, mask: &Tensor, value: f32) -> candle::Result<Tensor> {
    // Check if shapes are different
    if input.shape() != mask.shape() {
        // Attempt to broadcast the mask to the input shape
        let broadcasted_mask = mask.broadcast_as(input.shape())?.to_dtype(DType::I64)?;
        let fill_tensor =
            Tensor::full(value, input.shape(), input.device())?.to_dtype(DType::I64)?;
        t!("input", input);
        t!("broadcasted_mask", broadcasted_mask);
        t!("fill_tensor", fill_tensor);

        // FIX THIS could not foward: Cuda(UnexpectedDType { msg: "where conditions should be u8/u32/i64", expected: U32, got: F32 })
        // FIX THIS could not foward: Cuda(UnexpectedDType { msg: "where conditions should be u8/u32/i64", expected: U32, got: F32 })
        // FIX THIS could not foward: Cuda(UnexpectedDType { msg: "where conditions should be u8/u32/i64", expected: U32, got: F32 })
        // FIX THIS could not foward: Cuda(UnexpectedDType { msg: "where conditions should be u8/u32/i64", expected: U32, got: F32 })
        // FIX THIS could not foward: Cuda(UnexpectedDType { msg: "where conditions should be u8/u32/i64", expected: U32, got: F32 })
        // FIX THIS could not foward: Cuda(UnexpectedDType { msg: "where conditions should be u8/u32/i64", expected: U32, got: F32 })
        input.where_cond(&broadcasted_mask, &fill_tensor)
    } else {
        // Shapes are the same, proceed as before
        let fill_tensor = Tensor::full(value, input.shape(), input.device())?;
        input.where_cond(mask, &fill_tensor)
    }
}
*/

fn masked_fill(target: &Tensor, mask: &Tensor, value: f32) -> candle::Result<Tensor> {
    todo!()
}

struct XSoftmax {}

impl XSoftmax {
    pub fn apply(input: &Tensor, mask: &Tensor, dim: D, device: &Device) -> candle::Result<Tensor> {
        // t!("input", input);
        t!("attention_mask @ XSoftmax apply", mask);
        // NOTE: At the time of this writing, candle does not have a logical-not operator.
        let mut rmask = mask.broadcast_as(input.shape())?.to_dtype(DType::F32)?;
        t!("rmask", rmask);
        rmask = rmask
            .broadcast_lt(&Tensor::new(&[1.0 as f32], device)?)?
            .to_dtype(DType::U8)?;
        t!("rmask", rmask);

        // masked fill?
        let min_value_tensor = Tensor::new(&[f32::MIN], device)?.broadcast_as(input.shape())?;
        let mut output = rmask.where_cond(&min_value_tensor, &input)?;

        t!("output", output);
        output = candle_nn::ops::softmax(&output, dim)?;
        t!("output", output);

        let t_zeroes = Tensor::new(&[0f32], device)?.broadcast_as(input.shape())?;
        output = rmask.where_cond(&t_zeroes, &output)?;
        t!("output", output);

        Ok(output)
    }
}

// pub struct DebertaV2DisentangledSelfAttention<'a> {
pub struct DebertaV2DisentangledSelfAttention {
    pub config: Config,
    pub num_attention_heads: usize,
    pub attention_head_size: usize,
    pub query_proj: candle_nn::Linear,
    pub key_proj: candle_nn::Linear,
    pub value_proj: candle_nn::Linear,
    pub dropout: StableDropout,
    pub device: Device,
    pub relative_attention: bool,
    pub pos_dropout: Option<StableDropout>,
    pub position_buckets: isize,
    pub max_relative_positions: isize,
    pub pos_ebd_size: isize,
    pub share_att_key: bool,
    pub pos_key_proj: Option<candle_nn::Linear>,
    pub pos_query_proj: Option<candle_nn::Linear>,
}

// impl<'a> DebertaV2DisentangledSelfAttention<'a> {
impl DebertaV2DisentangledSelfAttention {
    // pub fn load(vb: VarBuilder<'a>, config: &Config) -> candle::Result<Self> {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let config = config.clone();
        let vb = vb.clone();

        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(candle::Error::Msg(format!(
                "The hidden size {} is not a multiple of the number of attention heads {}",
                config.hidden_size, config.num_attention_heads
            )));
        }

        let num_attention_heads = config.num_attention_heads;

        let attention_head_size = config
            .attention_head_size
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        let all_head_size = num_attention_heads * attention_head_size;

        let query_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("query_proj"))?;
        let key_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("key_proj"))?;
        let value_proj = candle_nn::linear(config.hidden_size, all_head_size, vb.pp("value_proj"))?;

        let share_att_key = config.share_att_key.unwrap_or(false);
        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions;

        let mut pos_ebd_size: isize = 0;
        let mut position_buckets = config.position_buckets.unwrap_or(-1);
        let mut pos_dropout: Option<StableDropout> = None;
        let mut pos_key_proj: Option<candle_nn::Linear> = None;
        let mut pos_query_proj: Option<candle_nn::Linear> = None;

        if relative_attention {
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings as isize;
            }
            pos_ebd_size = max_relative_positions;
            if position_buckets > 0 {
                pos_ebd_size = position_buckets
            }

            pos_dropout = Some(StableDropout::new(config.hidden_dropout_prob));

            if !share_att_key {
                if config.pos_att_type.contains(&"c2p".to_string()) {
                    pos_key_proj = Some(candle_nn::linear(
                        config.hidden_size,
                        all_head_size,
                        vb.pp("pos_key_proj"),
                    )?);
                }
                if config.pos_att_type.contains(&"p2c".to_string()) {
                    pos_query_proj = Some(candle_nn::linear(
                        config.hidden_size,
                        all_head_size,
                        vb.pp("pos_query_proj"),
                    )?);
                }
            }
        }

        let dropout = StableDropout::new(config.attention_probs_dropout_prob);
        let device = vb.device().clone();

        Ok(Self {
            config,
            num_attention_heads,
            attention_head_size,
            query_proj,
            key_proj,
            value_proj,
            dropout,
            device,
            relative_attention,
            pos_dropout,
            position_buckets,
            max_relative_positions,
            pos_ebd_size,
            share_att_key,
            pos_key_proj,
            pos_query_proj,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> candle::Result<Tensor> {
        let query_states = match query_states {
            Some(qs) => qs,
            None => hidden_states,
        };

        t!("attention_mask @ DSA forward", attention_mask);
        // println!(
        //     "query_states: {:?}\n{}",
        //     query_states.dims(),
        //     query_states.to_string()
        // );

        let query_layer = self.transpose_for_scores(&self.query_proj.forward(query_states)?)?;
        // println!(
        //     "query_layer: {:?}\n{}",
        //     query_layer.dims(),
        //     query_layer.to_string()
        // );
        let key_layer = self.transpose_for_scores(&self.key_proj.forward(query_states)?)?;
        // println!(
        //     "key_layer: {:?}\n{}",
        //     key_layer.dims(),
        //     key_layer.to_string()
        // );
        let value_layer = self.transpose_for_scores(&self.value_proj.forward(query_states)?)?;
        // println!(
        //     "value_layer: {:?}\n{}",
        //     value_layer.dims(),
        //     value_layer.to_string()
        // );

        let mut rel_att: Option<Tensor> = None;

        let mut scale_factor: usize = 1;

        if self.config.pos_att_type.contains(&"c2p".to_string()) {
            scale_factor += 1;
        }

        if self.config.pos_att_type.contains(&"p2c".to_string()) {
            scale_factor += 1;
        }

        let scale = {
            let q_size = query_layer.dims().last().unwrap();
            Tensor::new(&[(q_size * scale_factor) as f32], &self.device)?.sqrt()?
        };

        let mut attention_scores: Tensor = {
            let key_layer_transposed = key_layer.transpose(D::Minus1, D::Minus2)?;
            // println!(
            //     "key_layer_transposed: {:?}\n{}",
            //     key_layer_transposed.dims(),
            //     key_layer_transposed.to_string()
            // );
            let div = key_layer_transposed
                .broadcast_div(scale.to_dtype(query_layer.dtype())?.as_ref())?;
            // println!("div: {:?}\n{}", div.dims(), div.to_string());
            query_layer.matmul(&div)?
        };

        // println!(
        //     "attention_scores: {:?}\n{}",
        //     attention_scores.dims(),
        //     attention_scores.to_string()
        // );

        if self.relative_attention {
            let rel_embeddings = self
                .pos_dropout
                .as_ref()
                .ok_or(candle::Error::Msg(
                    "relative_attention requires pos_dropout".to_string(),
                ))?
                .forward(rel_embeddings)?
                .unwrap();

            rel_att = Some(self.disentangled_attention_bias(
                query_layer,
                key_layer,
                relative_pos,
                rel_embeddings,
                scale_factor,
            )?);
        }

        if rel_att.is_some() {
            attention_scores = attention_scores.broadcast_add(&rel_att.unwrap())?;
        }

        t!("attention_scores", attention_scores);

        attention_scores = attention_scores.reshape((
            (),
            self.num_attention_heads,
            attention_scores.dim(D::Minus2)?,
            attention_scores.dim(D::Minus1)?,
        ))?;

        t!("attention_scores", attention_scores);

        let mut attention_probs =
            XSoftmax::apply(&attention_scores, &attention_mask, D::Minus1, &self.device)?;

        t!("attention_probs", attention_probs);

        attention_probs =
            self.dropout
                .forward(Some(&attention_probs))?
                .ok_or(candle::Error::Msg(
                    "Dropout did not return a value".to_string(),
                ))?;

        t!("attention_probs", attention_probs);

        let mut context_layer = attention_probs
            .reshape((
                (),
                attention_probs.dim(D::Minus2)?,
                attention_probs.dim(D::Minus1)?,
            ))?
            .matmul(&value_layer)?;

        t!("context_layer", context_layer);

        context_layer = context_layer
            .reshape((
                (),
                self.num_attention_heads,
                context_layer.dim(D::Minus2)?,
                context_layer.dim(D::Minus1)?,
            ))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        t!("context_layer", context_layer);

        // let new_context_layer_shape = {
        //     let g = (1,2,3);
        //     let f = context_layer.reshape()
        //     let shape = context_layer.dims();
        //     let f = context_layer.shape();
        //     let new_shape: Vec<D> = shape[(..shape.len() - 2).into()].to_vec();
        //     new_shape.push(D::Minus1 as isize);
        // };
        // macro_rules! context_resize_tuple {
        //     ($slice:expr) => {{
        //         let len = $slice.len();
        //         match len {
        //             2 => ((),),
        //             3 => ($slice[0], ()),
        //             4 => ($slice[0], $slice[1], ()),
        //             5 => ($slice[0], $slice[1], $slice[2], ()),
        //             _ => panic!("Slice length must be between 2 and 5"),
        //         }
        //     }};
        // }

        // To replicate the following lines in the Python code:
        //   new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        //   context_layer = context_layer.view(new_context_layer_shape)
        let dims = context_layer.dims();

        context_layer = match dims.len() {
            2 => context_layer.reshape(())?,
            3 => context_layer.reshape((dims[0], ()))?,
            4 => context_layer.reshape((dims[0], dims[1], ()))?,
            5 => context_layer.reshape((dims[0], dims[1], dims[2], ()))?,
            _ => {
                return Err(candle::Error::Msg(format!(
                    "Invalid shape for DisentabgledSelfAttention context layer: {:?}",
                    dims
                )))
            }
        };

        t!("context_layer", context_layer);

        Ok(context_layer)
    }

    // fn transpose_for_scores(&self, x: &Tensor, attention_heads:)
    fn transpose_for_scores(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let mut xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs = xs.contiguous()?;
        xs.squeeze(0)
    }

    fn disentangled_attention_bias(
        &self,
        query_layer: Tensor,
        key_layer: Tensor,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Tensor,
        scale_factor: usize,
    ) -> candle::Result<Tensor> {
        let mut relative_pos: Tensor = if relative_pos.is_none() {
            let q = query_layer.dim(D::Minus2)?;
            build_relative_position(
                q,
                key_layer.dim(D::Minus2).unwrap(),
                &self.device,
                Some(self.position_buckets),
                Some(self.max_relative_positions),
            )?
        } else {
            relative_pos.cloned().unwrap()
        };

        t!("relative_pos", relative_pos);

        relative_pos = match relative_pos.dims().len() {
            2 => relative_pos.unsqueeze(0)?.unsqueeze(0)?,
            3 => relative_pos.unsqueeze(1)?,
            other => {
                return Err(candle::Error::Msg(format!(
                    "Relative position ids must be of dim 2 or 3 or 4. Got dim of size {other}"
                )))
            }
        };

        t!("relative_pos", relative_pos);

        let att_span = self.pos_ebd_size;

        relative_pos = relative_pos.to_dtype(DType::I64)?;

        // t!("relative_pos", relative_pos);

        // println!(
        //     "relative_pos: {:?}\n{}",
        //     relative_pos.dims(),
        //     relative_pos.to_string()
        // );

        let rel_embeddings = {
            let sliced = rel_embeddings.narrow(0, 0, (att_span * 2) as usize)?;
            sliced.unsqueeze(0)?
        };

        t!("rel_embeddings", rel_embeddings);

        let mut pos_query_layer: Option<Tensor> = None;
        let mut pos_key_layer: Option<Tensor> = None;

        let repeat_with = query_layer.dim(0)? / self.num_attention_heads;
        if self.share_att_key {
            let qproj = self.query_proj.forward(&rel_embeddings)?;
            t!("qproj", qproj);
            let transposed = self.transpose_for_scores(&qproj)?;
            t!("transposed", transposed);
            pos_query_layer = Some(transposed.repeat(repeat_with)?);
            t!("pos_query_layer", pos_query_layer.as_ref().unwrap());

            let kproj = self.key_proj.forward(&rel_embeddings)?;
            t!("kproj", kproj);
            let transposed = self.transpose_for_scores(&kproj)?;
            t!("transposd", transposed);
            pos_key_layer = Some(transposed.repeat(repeat_with)?);
            t!("pos_key_layer", pos_key_layer.as_ref().unwrap());
        } else {
            if self.config.pos_att_type.contains(&"c2p".to_string()) {
                let kproj = self
                    .pos_key_proj
                    .as_ref()
                    .ok_or(candle::Error::Msg(
                        "Need a pos_key_proj when share_att_key is false or not specified"
                            .to_string(),
                    ))?
                    .forward(&rel_embeddings)?;
                let transposed = self.transpose_for_scores(&kproj)?;
                pos_key_layer = Some(transposed.repeat(repeat_with)?);
            }
            if self.config.pos_att_type.contains(&"p2c".to_string()) {
                let qproj = self
                    .pos_query_proj
                    .as_ref()
                    .ok_or(candle::Error::Msg(
                        "Need a pos_query_proj when share_att_key is false or not specified"
                            .to_string(),
                    ))?
                    .forward(&rel_embeddings)?;
                let transposed = self.transpose_for_scores(&qproj)?;
                pos_query_layer = Some(transposed.repeat(repeat_with)?);
            }
        }

        t!("pos_key_layer", pos_key_layer.as_ref().unwrap());
        t!("pos_query_layer", pos_query_layer.as_ref().unwrap());

        let mut score = Tensor::new(&[0 as f32], &self.device)?;

        if self.config.pos_att_type.contains(&"c2p".to_string()) {
            let pos_key_layer = pos_key_layer.ok_or(candle::Error::Msg(
                "content to position without pos_key_layer".to_string(),
            ))?;

            let scale = {
                let layer_size = pos_key_layer.dim(D::Minus1)?;
                Tensor::new(&[(layer_size * scale_factor) as f32], &self.device)?.sqrt()?
            };

            t!("scale", scale);

            let mut c2p_att = {
                let transposed = pos_key_layer.transpose(D::Minus1, D::Minus2)?;
                query_layer.matmul(&transposed)?
            };

            t!("c2p_att", c2p_att);

            let c2p_pos = {
                // let att_span_t = Tensor::new(&[att_span as f32], &self.device)?;
                let att_span_t = Tensor::new(&[att_span as i64], &self.device)?;
                let rel_pos_plus_att_span = relative_pos.broadcast_add(&att_span_t)?;
                rel_pos_plus_att_span.clamp(0 as f32, (att_span * 2 - 1) as f32)?
            };

            t!("c2p_pos", c2p_pos);

            c2p_att = {
                let gather_idx = c2p_pos
                    .squeeze(0)?
                    .expand(&[
                        query_layer.dim(0)?,
                        query_layer.dim(1)?,
                        relative_pos.dim(D::Minus1)?,
                    ])?
                    .contiguous()?;

                t!("gather_idx", gather_idx);

                c2p_att.gather(&gather_idx, D::Minus1)?
            };

            t!("c2p_att", c2p_att);

            score = score.broadcast_add(
                &c2p_att.broadcast_div(scale.to_dtype(c2p_att.dtype())?.as_ref())?,
            )?;

            t!("score", score);
        }

        if self.config.pos_att_type.contains(&"p2c".to_string()) {
            let pos_query_layer = pos_query_layer.ok_or(candle::Error::Msg(
                "content to position without pos_key_layer".to_string(),
            ))?;

            let scale = {
                let layer_size = pos_query_layer.dim(D::Minus1)?;
                Tensor::new(&[(layer_size * scale_factor) as f32], &self.device)?.sqrt()?
            };
            t!("scale", scale);

            let r_pos = {
                if key_layer.dim(D::Minus2)? != query_layer.dim(D::Minus2)? {
                    build_relative_position(
                        key_layer.dim(D::Minus2)?,
                        key_layer.dim(D::Minus2)?,
                        &self.device,
                        Some(self.position_buckets),
                        Some(self.max_relative_positions),
                    )?
                    .unsqueeze(0)?
                } else {
                    relative_pos
                }
            };

            t!("r_pos", r_pos);

            let p2c_pos = {
                let att_span_t = Tensor::new(&[att_span as f32], &self.device)?;
                // let to_clamp = r_pos.neg()?.broadcast_add(&att_span_t)?;
                let to_clamp = r_pos
                    .to_dtype(DType::F32)?
                    .neg()?
                    .broadcast_add(&att_span_t)?;

                t!("to_clamp", to_clamp);
                to_clamp.clamp(0f32, (att_span * 2 - 1) as f32)?
            };

            t!("p2c_pos", p2c_pos);

            let p2c_att = {
                let transposed = pos_query_layer.transpose(D::Minus1, D::Minus2)?;
                let bmm = key_layer.matmul(&transposed)?;
                let gather_idx = p2c_pos
                    .squeeze(0)?
                    .expand(&[
                        query_layer.dim(0)?,
                        key_layer.dim(D::Minus2)?,
                        key_layer.dim(D::Minus2)?,
                    ])?
                    .contiguous()?
                    .to_dtype(DType::U32)?;
                t!("gather_idx", gather_idx);
                bmm.gather(&gather_idx, D::Minus1)?
                    .transpose(D::Minus1, D::Minus2)?
            };

            t!("p2c_att", p2c_att);

            score =
                score.broadcast_add(&p2c_att.broadcast_div(&scale.to_dtype(p2c_att.dtype())?)?)?;

            t!("score", score);
        }
        // println!(
        //     "rel_embeddings: {:?}\n{}",
        //     rel_embeddings.dims(),
        //     rel_embeddings.to_string()
        // );

        // todo!()
        Ok(score)
    }
}

// pub struct DebertaV2Attention<'a> {
pub struct DebertaV2Attention {
    // dsa: DebertaV2DisentangledSelfAttention<'a>,
    dsa: DebertaV2DisentangledSelfAttention,
    output: DebertaV2SelfOutput,
    config: Config,
}

// impl<'a> DebertaV2Attention<'a> {
impl DebertaV2Attention {
    // pub fn load(vb: VarBuilder<'a>, config: &Config) -> candle::Result<Self> {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let dsa = DebertaV2DisentangledSelfAttention::load(vb.pp("attention.self"), config)?;
        let output = DebertaV2SelfOutput::load(vb.pp("attention.output"), config)?;
        Ok(Self {
            dsa,
            output,
            config: config.clone(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> candle::Result<Tensor> {
        // &self,
        // hidden_states: &Tensor,
        // attention_mask: &Tensor,
        // query_states: Option<&Tensor>,
        // relative_pos: Option<&Tensor>,
        // rel_embeddings: Option<&Tensor>,
        let self_output = self.dsa.forward(
            hidden_states,
            &attention_mask,
            query_states,
            relative_pos,
            rel_embeddings,
        )?;

        let mut query_states = query_states;
        if query_states.is_none() {
            query_states = Some(hidden_states)
        }

        Ok(self.output.forward(&self_output, &query_states.unwrap())?)
    }
}

pub struct DebertaV2SelfOutput {
    pub dense: candle_nn::Linear,
    pub layer_norm: LayerNorm,
    pub dropout: StableDropout,
}

impl DebertaV2SelfOutput {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let dense = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = StableDropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> candle::Result<Tensor> {
        let mut hidden_states = self.dense.forward(hidden_states)?;
        t!("hidden_states", hidden_states);
        hidden_states =
            self.dropout
                .forward(Some(&hidden_states))?
                .ok_or(candle::error::Error::Msg(
                    "DebertaV2SelfOuput dropout did not return a Tensor".to_string(),
                ))?;
        t!("hidden_states", hidden_states);
        hidden_states = {
            let to_norm = hidden_states.broadcast_add(input_tensor)?;
            self.layer_norm.forward(&to_norm)?
        };
        t!("hidden_states", hidden_states);
        Ok(hidden_states)
    }
}

pub struct DebertaV2Intermediate {
    pub dense: candle_nn::Linear,
    pub intermediate_act: HiddenActLayer,
}

impl DebertaV2Intermediate {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let dense = candle_nn::linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("intermediate.dense"),
        )?;
        let intermediate_act = HiddenActLayer::new(config.hidden_act);
        Ok(Self {
            dense,
            intermediate_act,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> candle::Result<Tensor> {
        // let mut hidden_states = hidden_states;
        let mut hidden_states = self.dense.forward(&hidden_states)?;
        t!("hidden_states", hidden_states);
        hidden_states = self.intermediate_act.forward(&hidden_states)?;
        t!("hidden_states", hidden_states);

        Ok(hidden_states)
    }
}

pub struct DebertaV2Output {
    pub dense: candle_nn::Linear,
    pub layer_norm: LayerNorm,
    pub dropout: StableDropout,
}

impl DebertaV2Output {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let dense = candle_nn::linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("output.dense"),
        )?;
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("output.LayerNorm"),
        )?;
        let dropout = StableDropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> candle::Result<Tensor> {
        let mut hidden_states = self.dense.forward(&hidden_states)?;
        t!("hidden_states", hidden_states);
        hidden_states =
            self.dropout
                .forward(Some(&hidden_states))?
                .ok_or(candle::error::Error::Msg(
                    "DebertaV2Ouptut did not receive a Tensor after dropout".to_string(),
                ))?;
        t!("hidden_states", hidden_states);
        hidden_states = {
            let to_norm = hidden_states.broadcast_add(input_tensor)?;
            self.layer_norm.forward(&to_norm)?
        };
        t!("hidden_states", hidden_states);
        Ok(hidden_states)
    }
}

pub struct DebertaV2Layer {
    pub attention: DebertaV2Attention,
    pub intermediate: DebertaV2Intermediate,
    pub output: DebertaV2Output,
}

impl DebertaV2Layer {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let attention = DebertaV2Attention::load(vb.clone(), config)?;
        let intermediate = DebertaV2Intermediate::load(vb.clone(), config)?;
        let output = DebertaV2Output::load(vb.clone(), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> candle::Result<Tensor> {
        let attention_output = self.attention.forward(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            rel_embeddings,
        )?;
        t!("attention_output", attention_output);

        let intermediate_output = self.intermediate.forward(&attention_output)?;

        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;

        Ok(layer_output)
    }

    // pub fn forward(&self) -> candle::Result<Tensor> {
    //     todo!()
    // }
}

pub struct ConvLayer {
    conv_act: String,
    conv: Conv1d,
    layer_norm: LayerNorm,
    dropout: StableDropout,
    config: Config,
}

impl ConvLayer {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let config = config.clone();
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let groups = config.conv_groups.unwrap_or(1);
        let conv_act: String = config.conv_act.clone().unwrap_or("tanh".to_string());

        // TODO: Check the defaults against the Python version.
        let mut conv_conf = Conv1dConfig::default();
        conv_conf.padding = (kernel_size - 1) / 2;
        conv_conf.groups = groups;

        let conv = conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            conv_conf,
            vb.pp("conv"),
        )?;

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = StableDropout::new(config.hidden_dropout_prob);

        Ok(Self {
            conv_act,
            conv,
            layer_norm,
            dropout,
            config,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual_states: &Tensor,
        input_mask: &Tensor,
    ) -> candle::Result<Tensor> {
        todo!("Need a model that contains a conv layer to test against.")
    }
}

pub struct DebertaV2Encoder {
    pub layer: Vec<DebertaV2Layer>,
    pub relative_attention: bool,
    pub max_relative_positions: isize,
    pub position_buckets: isize,
    pub rel_embeddings: Option<Embedding>,
    pub norm_rel_ebd: String,
    pub layer_norm: Option<LayerNorm>,
    pub conv: Option<ConvLayer>,
    pub gradient_checkpointing: bool,
    pub device: Device,
}

impl DebertaV2Encoder {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let layer = (0..config.num_hidden_layers)
            .map(|index| DebertaV2Layer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<candle::Result<Vec<_>>>()?;

        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions;

        let position_buckets = match config.position_buckets {
            Some(ps) => ps,
            None => -1,
        };

        let mut rel_embeddings: Option<Embedding> = None;

        if relative_attention {
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings as isize;
            }

            let mut pos_ebd_size = max_relative_positions * 2;

            if position_buckets > 0 {
                pos_ebd_size = position_buckets * 2;
            }

            rel_embeddings = Some(embedding(
                pos_ebd_size as usize,
                config.hidden_size,
                vb.pp("rel_embeddings"),
            )?);
        }

        // NOTE: The Python code assumes that the config attribute "norm_rel_ebd" is an array of some kind, but most examples have it as a string.
        // So it might need to be updated at some point.
        let norm_rel_ebd = match config.norm_rel_ebd.as_ref() {
            Some(nre) => nre.trim().to_string(),
            None => "none".to_string(),
        };

        let layer_norm: Option<LayerNorm> = match norm_rel_ebd == "layer_norm" {
            true => Some(layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("LayerNorm"),
            )?),
            false => None,
        };

        let conv: Option<ConvLayer> = match config.conv_kernel_size.unwrap_or(0) > 0 {
            true => Some(ConvLayer::load(vb.pp("conv"), config)?),
            false => None,
        };

        Ok(Self {
            layer,
            relative_attention,
            max_relative_positions,
            position_buckets,
            rel_embeddings,
            norm_rel_ebd,
            layer_norm,
            conv,
            gradient_checkpointing: false,
            device: vb.device().clone(),
        })
    }
    // pub fn forward(
    //     &self,
    //     input_ids: Option<&Tensor>,
    //     token_type_ids: Option<&Tensor>,
    //     position_ids: Option<&Tensor>,
    //     mask: Option<&Tensor>,
    //     inputs_embeds: Option<&Tensor>,
    // ) -> candle::Result<Tensor> {
    //     todo!()
    // }
    //  def forward(
    //     self,
    //     hidden_states,
    //     attention_mask,
    //     output_hidden_states=True,
    //     output_attentions=False,
    //     query_states=None,
    //     relative_pos=None,
    //     return_dict=True,
    // ):
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> candle::Result<Tensor> {
        let input_mask = if attention_mask.dims().len() <= 2 {
            attention_mask.clone()
        } else {
            attention_mask
                .sum_keepdim(attention_mask.rank() - 2)?
                .gt(0.)?
        };

        t!("input_mask @ DebertaV2Encoder forward", input_mask);

        let attention_mask = self.get_attention_mask(attention_mask.clone())?;

        t!("attention_mask @ DebertaV2Encoder", attention_mask);

        let relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)?;

        // let next_kv = hidden_states;
        let mut next_kv: Tensor = hidden_states.clone();
        let rel_embeddings = self.get_rel_embedding()?;
        let mut output_states = next_kv.to_owned();

        // let mut query_states = query_states;
        let mut query_states: Option<Tensor> = query_states.cloned();

        for (i, layer_module) in self.layer.iter().enumerate() {
            // TODO: Ignoring output_hidden_states for now

            // NOTE: The original python code branches here if this model is being
            // used for training vs. inferencing. For now, we will only handle the
            // inferencing side of things

            // let mut output_states = layer_module.forward(
            output_states = layer_module.forward(
                next_kv.as_ref(),
                &attention_mask,
                query_states.as_ref(),
                relative_pos.as_ref(),
                rel_embeddings.as_ref(),
            )?;

            t!("output_states", output_states);

            if i == 0 && self.conv.is_some() {
                output_states = self.conv.as_ref().unwrap().forward(
                    hidden_states,
                    &output_states,
                    &input_mask,
                )?;
            }

            if query_states.is_some() {
                query_states = Some(output_states.clone());
            } else {
                next_kv = output_states.clone();
            }

            t!("next_kv", next_kv);
        }

        t!("output_states final", output_states);
        Ok(output_states)
    }

    // TEMP: Verified
    fn get_attention_mask(&self, mut attention_mask: Tensor) -> candle::Result<Tensor> {
        if attention_mask.dims().len() <= 2 {
            let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
            attention_mask = extended_attention_mask.broadcast_mul(
                &extended_attention_mask
                    .squeeze(D::Minus2)?
                    .unsqueeze(D::Minus1)?,
            )?;
        } else if attention_mask.dims().len() == 3 {
            attention_mask = attention_mask.unsqueeze(1)?;
        }

        Ok(attention_mask)
    }

    fn get_rel_pos(
        &self,
        hidden_states: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> candle::Result<Option<Tensor>> {
        if self.relative_attention && relative_pos.is_none() {
            let q = if let Some(query_states) = query_states {
                query_states.dim(D::Minus2)?
            } else {
                hidden_states.dim(D::Minus2)?
            };

            return Ok(Some(build_relative_position(
                q,
                hidden_states.dim(D::Minus2)?,
                &self.device,
                Some(self.position_buckets),
                Some(self.max_relative_positions),
            )?));
        }

        if relative_pos.is_some() {
            Ok(relative_pos.cloned())
        } else {
            Ok(None)
        }
    }
    fn get_rel_embedding(&self) -> candle::Result<Option<Tensor>> {
        // let rel_embeddings = self.rel_embeddings.unwrap().
        let mut rel_embeddings: Option<Tensor>;

        rel_embeddings = if self.relative_attention {
            Some(self.rel_embeddings.as_ref().unwrap().embeddings().clone())
        } else {
            None
        };

        if rel_embeddings.is_some() && self.norm_rel_ebd.contains("layer_norm") {
            rel_embeddings = Some(
                self.layer_norm
                    .as_ref()
                    .unwrap()
                    .forward(&rel_embeddings.unwrap())?,
            );
        };

        Ok(rel_embeddings)
    }

    // fn make_log_bucket_position(&self) -> candle::Result<Tensor> {
    //     todo!()
    // }
}

// pub struct DebertaV2Model<'vb> {
pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    z_steps: usize,
    config: Config,
    pub device: Device,
    // vb: VarBuilder<'vb>,
}

// impl<'vb> DebertaV2Model<'vb> {
impl DebertaV2Model {
    // pub fn load(vb: VarBuilder<'vb>, config: &Config) -> candle::Result<Self> {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let vb = vb.clone();
        let embeddings = DebertaV2Embeddings::load(vb.pp("embeddings"), config)?;
        let encoder = DebertaV2Encoder::load(vb.pp("encoder"), config)?;
        let z_steps: usize = 0;

        Ok(Self {
            embeddings,
            encoder,
            z_steps,
            config: config.clone(),
            device: vb.device().clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> candle::Result<Tensor> {
        let input_ids_shape = input_ids.shape();

        let attention_mask = match attention_mask {
            Some(mask) => mask,
            // None => Tensor::ones(input_ids_shape, DType::U32, self.vb.device())?,
            None => Tensor::ones(input_ids_shape, DType::I64, &self.device)?,
        };

        t!("attention_mask @ DebertaV2Model forward", attention_mask);

        let token_type_ids = match token_type_ids {
            Some(ids) => ids,
            // None => Tensor::zeros(input_ids_shape, DType::I64, self.vb.device())?,
            None => Tensor::zeros(input_ids_shape, DType::U32, &self.device)?,
        };

        t!("token_type_ids @ DebertaV2Model forward", token_type_ids);

        let embedding_output = self.embeddings.forward(
            Some(input_ids),
            Some(&token_type_ids),
            None,
            Some(&attention_mask),
            None,
        )?;

        // &self,
        // hidden_states: &Tensor,
        // attention_mask: &Tensor,
        // query_states: Option<Tensor>,
        // relative_pos: Option<Tensor>,
        let encoder_output =
            self.encoder
                .forward(&embedding_output, &attention_mask, None, None)?;

        t!("encoder_output @ DebertaV2Model", encoder_output);

        if self.z_steps > 1 {
            todo!("Copmlete DebertaV2Model forward() when z_steps > 1")
        }

        // Ok(embedding_output)
        Ok(encoder_output)
    }
}

pub struct SentencePiece {
    pub piece: String,
    pub id: u32,
    pub span: (u32, u32),
    pub is_special: bool,
}

#[derive(Debug)]
pub struct NERItem {
    entity: String,
    word: String,
    score: f32,
    start: u32,
    end: u32,
    index: usize,
}

pub struct DebertaV2NERModel {
    pub device: Device,
    id2label: Id2Label,
    deberta: DebertaV2Model,
    dropout: candle_nn::Dropout,
    classifier: candle_nn::Linear,
}

impl DebertaV2NERModel {
    pub fn load(
        vb: VarBuilder,
        config: &Config,
        id2label: Option<Id2Label>,
    ) -> candle::Result<Self> {
        let id2label: Id2Label = match &config.id2label {
            Some(i2l) => i2l.clone(),
            None => id2label.ok_or(candle::error::Error::Msg("Id2Label is either not present in the model configuration or not passed into DebertaV2NERModel::load as a parameter".to_string()))?
        };

        let deberta = DebertaV2Model::load(vb.clone(), &config)?;
        let dropout = candle_nn::Dropout::new(config.hidden_dropout_prob as f32);
        let classifier: candle_nn::Linear =
            candle_nn::linear_no_bias(config.hidden_size, 57, vb.root().pp("classifier"))?;

        Ok(Self {
            device: vb.device().clone(),
            deberta,
            id2label,
            dropout,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> candle::Result<Tensor> {
        let output = self
            .deberta
            .forward(input_ids, token_type_ids, attention_mask)?;
        let output = self.dropout.forward(&output, false)?;
        Ok(self.classifier.forward(&output)?)

        // for (idx, is_special) in self.special_tokens_mask.iter().enumerate() {
        //     if *is_special {
        //         continue;
        //     }

        //     let highest_score_idx = scores[idx]
        //         .iter()
        //         .enumerate()
        //         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        //         .map(|(index, _)| index)
        //         .unwrap();

        //     let label = config
        //         .id2label
        //         .as_ref()
        //         .unwrap()
        //         .get(&highest_score_idx)
        //         .unwrap()
        //         .clone();

        //     let (span, word) = match &encoded_tokens[idx] {
        //         SentencePieceToken::Ignore => ((0, 0), String::from("")),
        //         SentencePieceToken::Piece(piece) => {
        //             ((piece.span.0, piece.span.1), piece.piece.to_owned())
        //         }
        //     };

        //     values.push(NEREntity {
        //         label,
        //         word,
        //         score: scores[idx][highest_score_idx],
        //         start: span.0,
        //         end: span.1,
        //         index: idx,
        //     })
        // }
    }

    pub fn entities(
        &self,
        token_pieces: &Vec<SentencePiece>,
        token_type_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> candle::Result<Vec<Vec<NERItem>>> {
        let token_ids: Vec<u32> = token_pieces.iter().map(|piece| piece.id).collect();

        let input_ids: Tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;

        let logits = self.forward(&input_ids, token_type_ids, attention_mask)?;

        let maxes = logits.max_keepdim(D::Minus1)?;
        let shifted_exp = {
            let logits_minus_maxes = logits.broadcast_sub(&maxes)?;
            logits_minus_maxes.exp()?
        };
        let scores = {
            let sum = shifted_exp.sum_keepdim(D::Minus1)?;
            shifted_exp.broadcast_div(&sum)?
        };
        let scores = scores.squeeze(0)?.to_vec2::<f32>()?;
        let mut values: Vec<NERItem> = vec![];

        for (idx, piece) in token_pieces.iter().enumerate() {
            if piece.is_special {
                continue;
            }

            let highest_score_idx = scores[idx]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap();

            let label = self.id2label.get(&highest_score_idx).unwrap().clone();

            // let (span, word) = match &encoded_tokens[idx] {
            //     SentencePieceToken::Ignore => ((0, 0), String::from("")),
            //     SentencePieceToken::Piece(piece) => {
            //         ((piece.span.0, piece.span.1), piece.piece.to_owned())
            //     }
            // };

            values.push(NERItem {
                entity: label,
                word: piece.piece.clone(),
                score: scores[idx][highest_score_idx],
                start: piece.span.0,
                end: piece.span.1,
                index: idx,
            })
        }

        Ok(vec![values])
    }
}

pub(crate) fn build_relative_position(
    query_size: usize,
    key_size: usize,
    device: &Device,
    bucket_size: Option<isize>,
    max_position: Option<isize>,
) -> candle::Result<Tensor> {
    /*
    let q_ids = Tensor::arange(0, query_size as i64, &self.device)?;
    println!("q_ids: {:?}\n{}", q_ids, q_ids.to_string());

    // let q_ids_u = q_ids.unsqueeze(1)?;
    let q_ids_u = q_ids.unsqueeze(0)?;
    println!("q_ids_u: {:?}\n{}", q_ids_u, q_ids_u.to_string());

    let k_ids = Tensor::arange(0, key_size as i64, &self.device)?;
    println!("k_ids: {:?}\n{}", k_ids, k_ids.to_string());

    // let k_ids_u = k_ids.unsqueeze(0)?;
    let k_ids_u = k_ids.unsqueeze(D::Minus1)?;
    println!("k_ids_u: {:?}\n{}", k_ids_u, k_ids_u.to_string());

    let temp = q_ids_u.broadcast_sub(&k_ids_u)?;
    println!("temp: {:?}\n{}", temp, temp.to_string());

    let mut rel_pos_ids = q_ids.broadcast_sub(&k_ids)?;
    */

    let q_ids = Tensor::arange(0, query_size as i64, device)?.unsqueeze(0)?;
    let k_ids: Tensor = Tensor::arange(0, key_size as i64, device)?.unsqueeze(D::Minus1)?;
    // let mut rel_pos_ids = q_ids.broadcast_sub(&k_ids)?;
    let mut rel_pos_ids = k_ids.broadcast_sub(&q_ids)?;
    // println!(
    //     "real_pos_ids: {:?}\n{}",
    //     rel_pos_ids,
    //     rel_pos_ids.to_string()
    // );

    let bucket_size = bucket_size.unwrap_or(-1);
    let max_position = max_position.unwrap_or(-1);

    // rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    if bucket_size > 0 && max_position > 0 {
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position, device)?;
    }

    rel_pos_ids = rel_pos_ids.to_dtype(DType::I64)?;
    // rel_pos_ids = rel_pos_ids.slice_assign(&[0..query_size, ..], src)?;

    // let narrowed_rel_pos_ids = rel_pos_ids.narrow(0, 0, query_size)?;
    rel_pos_ids = rel_pos_ids.narrow(0, 0, query_size)?;

    // println!(
    //     "real_pos_ids narrow: {:?}\n{}",
    //     rel_pos_ids,
    //     rel_pos_ids.to_string()
    // );

    rel_pos_ids = rel_pos_ids.unsqueeze(0)?;

    // println!(
    //     "real_pos_ids unsqueeze: {:?}\n{}",
    //     rel_pos_ids,
    //     rel_pos_ids.to_string()
    // );

    Ok(rel_pos_ids)
}

pub(crate) fn make_log_bucket_position(
    relative_pos: Tensor,
    bucket_size: isize,
    max_position: isize,
    device: &Device,
) -> candle::Result<Tensor> {
    let sign = relative_pos
        .to_dtype(DType::F32)?
        .sign()?
        .to_dtype(DType::I64)?; // TODO: This cast might be unnecessary

    // println!("sign: {:?}\n{}", sign, sign.to_string());

    let mid = bucket_size / 2;

    let lt_mid = relative_pos.lt(mid as i64)?;
    let gt_neg_mid = relative_pos.gt(-mid as i64)?;

    let condition = lt_mid
        .to_dtype(candle::DType::F32)?
        .mul(&gt_neg_mid.to_dtype(candle::DType::F32)?)?;

    let condition_bool = condition.to_dtype(DType::U8)?;

    // println!(
    //     "condition_bool: {:?}\n{}",
    //     condition_bool,
    //     condition_bool.to_string()
    // );

    // let g = vec![(mid - 1)];

    let on_true = Tensor::new(&[(mid - 1) as u32], device)?.broadcast_as(relative_pos.shape())?;

    let on_true = on_true.to_dtype(relative_pos.dtype())?;

    let on_false = relative_pos
        .to_dtype(DType::F32)?
        .abs()?
        .to_dtype(DType::I64)?;

    // println!("on_true: {:?}\n{}", on_true, on_true.to_string());
    // println!("on_false: {:?}\n{}", on_false, on_false.to_string());

    let abs_pos = condition_bool.where_cond(&on_true, &on_false)?;
    /*
    torch.ceil(
        torch.log(abs_pos / mid) /
        torch.log(
            torch.tensor((max_position - 1) / mid)
        )
        * (mid - 1)
    ) + mid
     */

    let mid_as_tensor = Tensor::from_slice(&[mid as f32], (1,), device)?;

    let log_pos = {
        // let mid_as_tensor = Tensor::from_slice(&[mid as f32], (1,), &self.device)?;

        let first_log = abs_pos
            .to_dtype(DType::F32)?
            .broadcast_div(&mid_as_tensor)?
            .log()?;

        // println!("first_log: {:?}\n{}", first_log, first_log.to_string());

        let second_log = Tensor::from_slice(
            &[((max_position as f32 - 1.0) / mid as f32) as f32],
            (1,),
            device,
        )?
        .log()?;

        let first_div_second = first_log.broadcast_div(&second_log)?;

        let to_ceil = first_div_second
            .broadcast_mul(Tensor::from_slice(&[(mid - 1) as f32], (1,), device)?.as_ref())?;

        let ceil = to_ceil.ceil()?;

        // println!("ceil: {:?}\n{}", ceil, ceil.to_string());

        ceil.broadcast_add(&mid_as_tensor)?
    };

    // println!("log_pos: {:?}\n{}", log_pos, log_pos.to_string());

    let bucket_pos = {
        let abs_pos_lte_mid = abs_pos.to_dtype(DType::F32)?.broadcast_le(&mid_as_tensor)?;
        // println!(
        //     "abs_pos_lte_mid: {:?}\n{}",
        //     abs_pos_lte_mid,
        //     abs_pos_lte_mid.to_string()
        // );
        let relative_pos = relative_pos.to_dtype(relative_pos.dtype())?;
        // println!(
        //     "relative_pos: {:?}\n{}",
        //     relative_pos,
        //     relative_pos.to_string()
        // );
        let log_pos_mul_sign = log_pos.broadcast_mul(&sign.to_dtype(DType::F32)?)?;
        // println!(
        //     "log_pos_mul_sign: {:?}\n{}",
        //     log_pos_mul_sign,
        //     log_pos_mul_sign.to_string()
        // );

        abs_pos_lte_mid.where_cond(&relative_pos.to_dtype(DType::F32)?, &log_pos_mul_sign)?
    };
    Ok(bucket_pos)
}

// fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle::Result<Tensor> {
//     let shape = mask.shape();
//     let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
//     let m = mask.where_cond(&on_true, on_false)?;
//     Ok(m)
// }

// fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> candle::Result<Tensor> {
//     t!("on_false", on_false);
//     t!("mask", mask);
//     t!("on_true", on_true);
//     let shape = mask.shape();
//     // let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
//     let m = mask.where_cond(
//         &on_true.broadcast_as(shape.dims())?,
//         &on_false.broadcast_as(shape.dims())?,
//     )?;
//     t!("m", m);
//     Ok(m)
// }

// fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle::Result<Tensor> {
//     let shape = mask.shape();
//     let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
//     t!("on_true", on_true);
//     t!("on_false", on_false);
//     let m = mask.where_cond(&on_true, on_false)?;
//     Ok(m)
// }

/*

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle::Result<Tensor> {
    // Convert the mask to boolean and invert it
    // let rmask = mask.to_dtype(DType::Bool)?.logical_not()?;

    // Get the minimum value for the input's data type
    // let min_value = match input.dtype() {
    //     DType::F32 => std::f32::MIN,
    //     DType::F64 => std::f64::MIN,
    //     _ => return Err(candle::Error::Other("Unsupported data type".into())),
    // };

    // Create a tensor with the minimum value
    let min_value_tensor = Tensor::new(on_true, on_false.device())?;

    // Mask the input tensor
    let masked_input = on_false.where_cond(&mask, &min_value_tensor)?;

    // Apply the softmax operation
    let softmax_output = masked_input.softmax(dim)?;

    // Mask the softmax output, setting values to zero where the mask is `False`
    let zero_tensor = Tensor::zeros(&[], input.device())?;
    let final_output = softmax_output.where_cond(&rmask, &zero_tensor)?;

    Ok(final_output)

    // let shape = mask.shape();
    // let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    // let m = mask.where_cond(&on_true, on_false)?;
    // Ok(m)

    // // let shape = mask.shape();
    // let mut on_true = Tensor::new(on_true, on_false.device())?;
    // t!("on_false", on_false);
    // // t!("on_true", on_true);
    // // on_true = on_true.broadcast_as(shape.dims())?;
    // on_true = on_true.broadcast_as(on_false.dims())?;
    // let mask = mask.broadcast_as(on_false.shape())?;
    // t!("mask", mask);
    // t!("on_true", on_true);
    // let m = mask.where_cond(&on_true, on_false)?;
    // Ok(m)

    // let shape = mask.shape();

    // // Broadcast `on_true` to the shape of `on_false`
    // let on_true_tensor = Tensor::new(on_true, on_false.device())?.broadcast_as(on_false.shape())?;

    // // Broadcast the mask to the shape of `on_false`
    // let mask_broadcasted = mask.broadcast_as(on_false.shape())?;

    // let m = mask_broadcasted.where_cond(&on_true_tensor, on_false)?;
    // Ok(m)
}
*/

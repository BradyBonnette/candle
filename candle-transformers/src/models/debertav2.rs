use candle::{
    cuda::cudarc::driver::sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC, DType, Device,
    Module, Tensor,
};
use candle_nn::{
    conv1d, embedding, layer_norm, linear_b, Conv1d, Conv1dConfig, Embedding, LayerNorm, VarBuilder,
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

fn deserialize_pos_att_type<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = String::deserialize(deserializer)?;
    Ok(s.split('|').map(String::from).collect())
}

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

    pub fn forward(&self, x: Tensor) -> candle::Result<Tensor> {
        Ok(x)
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

        let position_ids = Tensor::arange(0.0f32, config.max_position_embeddings as f32, &device)?;

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

        let seq_length = input_shape.last().unwrap().to_owned();

        let position_ids = match position_ids {
            Some(p) => p.to_owned(),
            None => self.position_ids.narrow(0, 0, seq_length)?,
        };

        let token_type_ids = match token_type_ids {
            Some(t) => t.to_owned(),
            None => Tensor::zeros(input_shape, DType::I64, &self.device)?,
        };

        let input_embeds = match inputs_embeds {
            Some(e) => e.to_owned(),
            None => self.word_embeddings.forward(input_ids.unwrap())?,
        };

        let position_embeddings = match &self.position_embeddings {
            Some(emb) => emb.forward(&position_ids)?,
            None => Tensor::zeros_like(&input_embeds)?,
        };

        let mut embeddings = input_embeds;

        if self.config.position_biased_input {
            embeddings = embeddings.add(&position_embeddings)?;
        }

        if self.config.type_vocab_size > 0 {
            let token_type_embeddings = self.token_type_embeddings.as_ref().unwrap();
            let token_type_embeddings = token_type_embeddings.forward(&token_type_ids)?;
            embeddings = embeddings.add(&token_type_embeddings)?;
        }

        if self.embedding_size != self.config.hidden_size {
            embeddings = self.embed_proj.as_ref().unwrap().forward(&embeddings)?;
        }

        embeddings = self.layer_norm.forward(&embeddings)?;

        // TODO: Figure this out \/
        // if let Some(mut mask) = mask {
        //     if mask.dims() != embeddings.dims() {
        //         if mask.dims().len() == 4 {
        //             let mut mask = &mask.squeeze(1)?;
        //             mask = &mask.squeeze(1)?;
        //         }
        //         mask = &mask.unsqueeze(2)?;

        //         embeddings = embeddings.mul(mask)?;
        //     }

        //     // mask = mask.t

        // }

        embeddings = self.dropout.forward(embeddings)?;

        let g = embeddings.dims();
        println!("{}", embeddings.to_string());

        Ok(embeddings)
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
    // pub vb: VarBuilder<'a>,
    // pub vb: VarBuilder,
    pub dropout: StableDropout,
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

        let mut pos_ebd_size: isize;
        let mut position_buckets = config.position_buckets.unwrap_or(-1);
        let pos_dropout: Option<StableDropout>;
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

        Ok(Self {
            config,
            num_attention_heads,
            attention_head_size,
            query_proj,
            key_proj,
            value_proj,
            // vb,
            dropout,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        output_attentions: bool,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> candle::Result<Self> {
        let query_states = match query_states {
            Some(qs) => qs,
            None => hidden_states,
        };

        let query_layer = self.transpose_for_scores(&self.query_proj.forward(query_states)?)?;
        let key_layer = self.transpose_for_scores(&self.key_proj.forward(query_states)?)?;
        let value_layer = self.transpose_for_scores(&self.value_proj.forward(query_states)?)?;

        let rel_att: Option<Tensor> = None;

        let mut scale_factor: usize = 1;

        if self.config.pos_att_type.contains(&"c2p".to_string()) {
            scale_factor += 1;
        }

        if self.config.pos_att_type.contains(&"p2c".to_string()) {
            scale_factor += 1;
        }

        // let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
        let dims = query_layer.dims().last().unwrap();
        // let scale = Tensor::new(query_layer[..], self.vb.device())?;

        // let scale = {
        //     let dim_size = query_layer.dim(query_layer.dims() - 1)?;
        //     let dim_float = dim_size as f64;
        //     (dim_float * scale_factor).sqrt()
        // }?;

        todo!()
    }

    // fn transpose_for_scores(&self, x: &Tensor, attention_heads:)
    fn transpose_for_scores(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
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

    pub fn forward(&self) -> candle::Result<Self> {
        todo!()
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

    pub fn forward(&self) -> candle::Result<Self> {
        todo!()
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

    pub fn forward(&self) -> candle::Result<Tensor> {
        todo!()
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

    pub fn forward(&self) -> candle::Result<Tensor> {
        todo!()
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

    pub fn forward(&self) -> candle::Result<Tensor> {
        todo!()
    }
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

    pub fn forward(&self) -> candle::Result<Tensor> {
        todo!()
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
        todo!()
    }
}

pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    z_steps: usize,
    config: Config,
}

impl DebertaV2Model {
    pub fn load(vb: VarBuilder, config: &Config) -> candle::Result<Self> {
        let embeddings = DebertaV2Embeddings::load(vb.pp("embeddings"), config)?;
        let encoder = DebertaV2Encoder::load(vb.pp("encoder"), config)?;
        let z_steps: usize = 0;

        Ok(Self {
            embeddings,
            encoder,
            z_steps,
            config: config.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> candle::Result<Tensor> {
        // let embedding_output =
        //     self.embeddings
        //         .forward(Some(input_ids), Some(token_type_ids), None, None, None)?;

        // let _enter = self.span.enter();
        // let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        // let sequence_output = self.encoder.forward(&embedding_output)?;
        // Ok(sequence_output)
        todo!()
    }
}

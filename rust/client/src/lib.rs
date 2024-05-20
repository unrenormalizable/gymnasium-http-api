extern crate rand;
extern crate reqwest;
extern crate serde;
extern crate serde_json;

pub mod defs;
pub mod ui;

use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::ser::Serialize;
use serde_json::{to_value, Value};
use std::collections::HashMap;
use std::error::*;
use std::rc::Rc;
use value_extensions::*;

pub type Discrete = i64;
pub type Continous = f64;

#[derive(Debug)]
pub enum RenderFrame {
    Ansi(String),
    Rgb(usize, usize, Vec<u8>),
}

impl RenderFrame {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            RenderFrame::Ansi(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_rgb(&self) -> Option<(&usize, &usize, &Vec<u8>)> {
        match self {
            RenderFrame::Rgb(r, c, d) => Some((r, c, d)),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Transition {
    pub next_state: Discrete,
    pub probability: Continous,
    pub reward: f64,
    pub done: bool,
}

pub type Transitions = HashMap<(Discrete, Discrete), Vec<Transition>>;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct EpisodeEvent {
    pub s: Vec<Discrete>,
    pub r: f64,
}

pub trait Space {
    type Item;

    fn new(val: &Value) -> Self;

    fn action(val: &Value) -> Self::Item;

    fn observation(val: &Value) -> Self::Item;

    fn action_request(actions: &Self::Item) -> HashMap<&str, Value>;
}

#[derive(Debug)]
pub struct DiscreteSpace {
    pub n: Discrete,
}

impl Space for DiscreteSpace {
    type Item = Discrete;

    fn new(val: &Value) -> Self {
        let info = val["info"].as_object().unwrap();
        let name = info["name"].as_str().unwrap();
        if name != "Discrete" {
            panic!("name must be Discrete for Discrete spaces.")
        }

        Self {
            n: Discrete::from_value(&info["n"]).unwrap(),
        }
    }

    fn action(val: &Value) -> Discrete {
        Discrete::from_value(val).unwrap()
    }

    fn observation(val: &Value) -> Discrete {
        let ty = val["type"].as_str().unwrap();
        let data = val["data"].as_str().unwrap();

        let obs = deserialize_binary_stream::<Discrete>(ty, data);

        if obs.len() != 1 {
            panic!("For Discrete space: Expected only one observation.")
        }

        obs[0]
    }

    fn action_request(action: &Discrete) -> HashMap<&str, Value> {
        let mut req = HashMap::from([]);
        req.insert("action", to_value(action).unwrap());

        req
    }
}

pub trait FromCustom: Sized + core::fmt::Debug {
    fn from_value(val: &Value) -> Option<Self>;

    fn from_le_bytes(bytes: &[u8]) -> Result<Self, Box<dyn Error>>;
}

pub trait BoxSpaceElement: FromCustom + Serialize {}

impl BoxSpaceElement for Discrete {}

impl BoxSpaceElement for Continous {}

#[derive(Debug)]
pub struct BoxSpace<T: BoxSpaceElement> {
    pub shape: Vec<usize>,
    pub high: Vec<T>,
    pub low: Vec<T>,
}
use base64::prelude::*;
use flate2::read::ZlibDecoder;
use std::convert::TryInto;
use std::io::prelude::*;
use std::mem::size_of;

impl<T: BoxSpaceElement> Space for BoxSpace<T> {
    type Item = Vec<T>;

    fn new(val: &Value) -> Self {
        let info = val["info"].as_object().unwrap();
        let name = info["name"].as_str().unwrap();
        if name != "Box" {
            panic!("name must be Box for Box spaces.")
        }

        Self {
            shape: array_from_value::<usize>(&info["shape"]),
            high: array_from_value::<T>(&info["high"]),
            low: array_from_value::<T>(&info["low"]),
        }
    }

    fn action(val: &Value) -> Vec<T> {
        array_from_value::<T>(val)
    }

    fn observation(val: &Value) -> Vec<T> {
        let ty = val["type"].as_str().unwrap();
        let data = val["data"].as_str().unwrap();

        deserialize_binary_stream::<T>(ty, data)
    }

    fn action_request(action: &Self::Item) -> HashMap<&str, Value> {
        let action: Vec<_> = action.iter().collect();
        let mut req = HashMap::from([]);
        req.insert("action", to_value(action).unwrap());

        req
    }
}

#[derive(Debug)]
pub struct StepInfo<O: Space> {
    pub observation: O::Item,
    pub reward: f64,
    pub truncated: bool,
    pub terminated: bool,
    pub info: Value,
}

/// Create a gymnasium environment or get reference to an existing one.
/// NOTE: All APIs are sync for now as the server is expected to be local.
#[derive(Debug)]
pub struct Environment<O: Space, A: Space> {
    client: Client,
    api_url: String,
    instance_id: String,
    obs_space: O,
    act_space: A,
}

impl<O: Space, A: Space> Environment<O, A> {
    pub fn envs(api_url: &str) -> HashMap<String, String> {
        let client = Client::new(api_url);

        let url = client.make_api_url("");
        let val = client.http_get(&url);

        let obj = val["all_envs"]
            .as_object()
            .ok_or("No all_envs returned.")
            .unwrap();

        obj.into_iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect()
    }

    pub fn rc(self) -> Rc<Self> {
        Rc::new(self)
    }

    pub fn new(
        api_url: &str,
        env_name: &str,
        max_episode_steps: Option<Discrete>,
        auto_reset: Option<bool>,
        disable_env_checker: Option<bool>,
        kwargs: &[(&str, Value)],
    ) -> Self {
        let mut body = [("env_id", to_value(env_name).unwrap())]
            .into_iter()
            .collect::<HashMap<&str, Value>>();

        if let Some(max_episode_steps) = max_episode_steps {
            body.insert("max_episode_steps", to_value(max_episode_steps).unwrap());
        }

        if let Some(auto_reset) = auto_reset {
            body.insert("auto_reset", to_value(auto_reset).unwrap());
        }

        if let Some(disable_env_checker) = disable_env_checker {
            body.insert(
                "disable_env_checker",
                to_value(disable_env_checker).unwrap(),
            );
        }

        let kwargs = kwargs.iter().cloned().collect::<HashMap<&str, Value>>();
        body.insert("kwargs", to_value(kwargs).unwrap());

        let c = Client::new(api_url);
        let base_url = c.make_api_url("");
        let obj = c.http_post(&base_url, &body);
        let inst_id = obj["instance_id"].as_str().unwrap();

        Self::reference(api_url, inst_id)
    }

    pub fn reference(api_url: &str, instance_id: &str) -> Self {
        let client = Client::new(api_url);

        let url = client.make_api_url(&format!("{}/observation_space/", instance_id));
        let obj = client.http_get(&url);
        let obs_space = O::new(&obj);

        let url = client.make_api_url(&format!("{}/action_space/", instance_id));
        let obj = client.http_get(&url);
        let act_space = A::new(&obj);

        let env_api_url = client.make_api_url(&format!("{instance_id}/"));
        Self {
            client,
            api_url: env_api_url,
            instance_id: instance_id.to_string(),
            obs_space,
            act_space,
        }
    }

    pub fn client_base_url(&self) -> &str {
        &self.client.base_url
    }

    pub fn name(&self) -> String {
        let obj = self.client.http_get(&self.api_url);

        obj["id"].as_str().unwrap().to_string()
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// The Space object corresponding to valid actions, all valid actions should be contained with the space.
    /// For example, if the action space is of type Discrete and gives the value Discrete(2), this means there
    /// are two valid discrete actions: 0 & 1.
    /// Refer: https://gymnasium.farama.org/api/env/#gymnasium.Env.action_space
    pub fn action_space(&self) -> &A {
        &self.act_space
    }

    /// The Space object corresponding to valid observations, all valid observations should be contained with
    /// the space. For example, if the observation space is of type Box and the shape of the object is (4,),
    /// this denotes a valid observation will be an array of 4 numbers. We can check the box bounds as well with attributes.
    /// Refer: https://gymnasium.farama.org/api/env/#gymnasium.Env.observation_space
    pub fn observation_space(&self) -> &O {
        &self.obs_space
    }

    pub fn action_space_sample(&self) -> A::Item {
        let url = self.make_api_url("action_space/sample/");
        let obj = self.client.http_get(&url);
        A::action(&obj["action"])
    }

    pub fn episode_samples(&self, count: usize, seed: Option<usize>) -> Vec<Vec<EpisodeEvent>> {
        let mut body = HashMap::from([("count", count.to_string())]);
        if let Some(seed) = seed {
            let _ = body.insert("seed", seed.to_string());
        }

        let url = self.make_api_url("episodes/");
        let obj = self.client.http_post(&url, &body);
        serde_json::from_value::<Vec<Vec<EpisodeEvent>>>(obj["episodes"].clone()).unwrap()
    }

    pub fn reset(&self, seed: Option<usize>) -> O::Item {
        let mut body = HashMap::from([]);
        if let Some(seed) = seed {
            let _ = body.insert("seed", seed.to_string());
        }

        let url = self.make_api_url("reset/");
        let obj = self.client.http_post(&url, &body);
        O::observation(&obj["observation"])
    }

    pub fn render(&self) -> RenderFrame {
        let url = self.make_api_url("render/");
        let obj = self.client.http_get(&url);

        let rf = &obj["render_frame"];
        if rf.is_string() {
            RenderFrame::Ansi(rf.as_str().unwrap().to_string())
        } else if rf.is_object() {
            let obj = rf.as_object().unwrap();
            let rows = obj["rows"].as_u64().unwrap() as usize;
            let cols = obj["cols"].as_u64().unwrap() as usize;
            let data = obj["data"].as_str().unwrap();

            let data = deserialize_binary_stream_to_bytes(data);

            RenderFrame::Rgb(rows, cols, data)
        } else {
            unimplemented!()
        }
    }

    pub fn step(&self, action: &A::Item) -> StepInfo<O> {
        let req = A::action_request(action);

        let url = self.make_api_url("step/");
        let obj = self.client.http_post(&url, &req);
        let observation = O::observation(&obj["observation"]);

        StepInfo {
            observation,
            reward: obj["reward"].as_f64().unwrap(),
            truncated: obj["truncated"].as_bool().unwrap(),
            terminated: obj["terminated"].as_bool().unwrap(),
            info: obj["info"].clone(),
        }
    }

    fn make_api_url(&self, path: &str) -> String {
        format!("{}{path}", self.api_url)
    }
}

pub fn transitions(env: &Environment<DiscreteSpace, DiscreteSpace>) -> Rc<Transitions> {
    let url = env.make_api_url("transitions/");
    let obj = env.client.http_get(&url);
    let obj = obj["transitions"].as_object().unwrap();

    let mut transitions: Transitions = HashMap::new();

    let n_s = env.observation_space().n;
    let n_a = env.action_space().n;
    for s in 0..n_s {
        let s_trans = obj[&s.to_string()].as_object().unwrap();
        for a in 0..n_a {
            let a_trans = s_trans[&a.to_string()].as_array().unwrap();
            let ts = a_trans
                .iter()
                .map(|t| {
                    let t = t.as_array().unwrap();
                    Transition {
                        probability: Continous::from_value(&t[0]).unwrap(),
                        next_state: Discrete::from_value(&t[1]).unwrap(),
                        reward: Continous::from_value(&t[2]).unwrap(),
                        done: t[3].as_bool().unwrap(),
                    }
                })
                .collect();

            transitions.insert((s, a), ts);
        }
    }

    Rc::new(transitions)
}

#[derive(Debug)]
pub struct Client {
    base_url: String,
    api_url: String,
    client: reqwest::blocking::Client,
}

/// NOTE: Retaining sync implementations now as the scenario is only single threaded.
impl Client {
    pub fn new(base_url: &str) -> Self {
        let mut base_url = base_url.replace("//localhost:", "//127.0.0.1:");
        if base_url.ends_with('/') {
            _ = base_url.remove(base_url.len() - 1);
        }

        let api_url = format!("{base_url}/v1/envs/");

        Self {
            base_url,
            api_url,
            client: reqwest::blocking::Client::builder().build().unwrap(),
        }
    }

    pub fn make_api_url(&self, path: &str) -> String {
        format!("{}{}", self.api_url, path)
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn http_get(&self, url: &str) -> Value {
        let res = self
            .client
            .get(url)
            .headers(Self::construct_common_headers())
            .send();
        res.unwrap().json::<Value>().unwrap()
    }

    fn http_post<T: Serialize>(&self, url: &str, body: &HashMap<&str, T>) -> Value {
        let res = self
            .client
            .post(url)
            .headers(Self::construct_common_headers())
            .json(body)
            .send();
        res.unwrap().json::<Value>().unwrap()
    }

    fn construct_common_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }
}

mod value_extensions {
    use super::*;
    use serde_json::Value;

    impl FromCustom for Discrete {
        fn from_value(val: &Value) -> Option<Self> {
            val.as_u64().map(|x| x as Self)
        }

        fn from_le_bytes(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
            let x = bytes.try_into()?;
            Ok(Self::from_le_bytes(x))
        }
    }

    impl FromCustom for Continous {
        fn from_value(val: &Value) -> Option<Self> {
            val.as_f64().map(|x| x as Self)
        }

        fn from_le_bytes(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
            let x = bytes.try_into()?;
            Ok(Self::from_le_bytes(x))
        }
    }

    impl FromCustom for usize {
        fn from_value(val: &Value) -> Option<Self> {
            val.as_u64().map(|x| x as Self)
        }

        fn from_le_bytes(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
            let x = bytes.try_into()?;
            Ok(Self::from_le_bytes(x))
        }
    }

    pub fn array_from_value<T: FromCustom>(val: &Value) -> Vec<T> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|x| T::from_value(x).unwrap())
            .collect()
    }

    fn map_py_type_name_to_rust(name: &str) -> &str {
        match name {
            "int32" => "i32",
            "int64" => "i64",
            "float32" => "f32",
            "float64" => "f64",
            _ => "unknown",
        }
    }

    pub fn deserialize_binary_stream_to_bytes(data: &str) -> Vec<u8> {
        let data = BASE64_STANDARD.decode(data).unwrap();
        let mut dec = ZlibDecoder::new(&data[..]);
        let mut data = Vec::new();
        dec.read_to_end(&mut data).unwrap();

        data
    }

    pub fn deserialize_binary_stream<T: FromCustom>(ty: &str, data: &str) -> Vec<T> {
        if std::any::type_name::<T>() != map_py_type_name_to_rust(ty) {
            panic!("Mismatch in types. Ensure client and server have same types.")
        }

        let data = deserialize_binary_stream_to_bytes(data);

        if data.len() % size_of::<T>() != 0 {
            panic!("Recieved binary stream not in multiple of expected chunks.")
        }

        data.chunks_exact(size_of::<T>())
            .map(|chunk| T::from_le_bytes(chunk).unwrap())
            .collect()
    }
}

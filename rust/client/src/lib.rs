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
use std::rc::Rc;
use value_extensions::*;

pub type Discrete = i64;
pub type Continous = f64;

#[derive(Debug)]
pub enum RenderFrame {
    Ansi(String),
    Rgb(usize, usize, String),
}

impl RenderFrame {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            RenderFrame::Ansi(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_rgb(&self) -> Option<(&usize, &usize, &String)> {
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
    type T;
    type Item;

    fn new(val: &Value) -> Self;

    fn action(val: &Value) -> Self::Item;

    fn observation(vals: &[Value]) -> Self::Item;

    fn action_request(actions: &Self::Item) -> HashMap<&str, Value>;
}

#[derive(Debug)]
pub struct DiscreteSpace {
    pub n: Discrete,
}

impl Space for DiscreteSpace {
    type T = Discrete;
    type Item = Discrete;

    fn new(val: &Value) -> Self {
        let info = val["info"].as_object().unwrap();
        let name = info["name"].as_str().unwrap();
        if name != "Discrete" {
            panic!("name must be Discrete for Discrete spaces.")
        }

        Self {
            n: info["n"].as_i64().unwrap(),
        }
    }

    fn action(val: &Value) -> Discrete {
        val.as_i64().unwrap()
    }

    fn observation(vals: &[Value]) -> Discrete {
        if vals.len() != 1 {
            panic!("For Discrete space: Expected only one observation.")
        }

        vals[0].as_i64().unwrap()
    }

    fn action_request(action: &Discrete) -> HashMap<&str, Value> {
        let mut req = HashMap::from([]);
        req.insert("action", to_value(action).unwrap());

        req
    }
}

#[derive(Debug)]
pub struct BoxSpace {
    pub shape: Vec<Discrete>,
    pub high: Vec<Continous>,
    pub low: Vec<Continous>,
}

impl Space for BoxSpace {
    type T = Continous;
    type Item = Vec<Continous>;

    fn new(val: &Value) -> Self {
        let info = val["info"].as_object().unwrap();
        let name = info["name"].as_str().unwrap();
        if name != "Box" {
            panic!("name must be Box for Box spaces.")
        }

        Self {
            shape: as_discrete_item_vec(&info["shape"]),
            high: as_continous_item_vec(&info["high"]),
            low: as_continous_item_vec(&info["low"]),
        }
    }

    fn action(val: &Value) -> Vec<Continous> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>()
    }

    fn observation(vals: &[Value]) -> Vec<Continous> {
        vals.iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>()
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
        let obs = obj["observation"].as_array().unwrap();
        O::observation(obs)
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
            let data = obj["data"].as_str().unwrap().to_string();

            RenderFrame::Rgb(rows, cols, data)
        } else {
            unimplemented!()
        }
    }

    pub fn step(&self, action: &A::Item) -> StepInfo<O> {
        let req = A::action_request(action);

        let url = self.make_api_url("step/");
        let obj = self.client.http_post(&url, &req);
        let observation = obj["observation"].as_array().unwrap();
        let observation = O::observation(observation);

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
                        probability: t[0].as_f64().unwrap() as Continous,
                        next_state: t[1].as_i64().unwrap() as Discrete,
                        reward: t[2].as_f64().unwrap(),
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

    pub fn as_discrete_item_vec(val: &Value) -> Vec<Discrete> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as Discrete)
            .collect::<Vec<_>>()
    }

    pub fn as_continous_item_vec(val: &Value) -> Vec<Continous> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as Continous)
            .collect::<Vec<_>>()
    }
}

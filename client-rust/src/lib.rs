extern crate rand;
extern crate reqwest;
extern crate serde;
extern crate serde_json;

use reqwest::{
    blocking::*,
    header::{HeaderMap, HeaderValue, CONTENT_TYPE},
};
use serde::ser::Serialize;
use serde_json::{to_value, Map, Value};
use std::collections::HashMap;
use value_extensions::*;

#[derive(Debug, Clone)]
pub struct GymClient {
    base_uri: String,
    client: Client,
}

// TODO: Consider nuking this.
#[derive(Debug, Clone, Copy)]
pub enum ObsActSpaceItem {
    Discrete(Discrete),
    Continous(Continous),
}

impl ObsActSpaceItem {
    pub fn discrete_value(&self) -> Discrete {
        match self {
            Self::Discrete(n) => *n,
            _ => panic!("Is not a Discrete item"),
        }
    }

    pub fn box_value(&self) -> Continous {
        match self {
            Self::Continous(n) => *n,
            _ => panic!("Is not a Discrete item"),
        }
    }
}

pub type Discrete = i32;
pub type Continous = f64;

#[derive(Debug, Clone)]
pub enum ObsActSpace {
    /// Refer: https://www.gymlibrary.dev/api/spaces/#discrete
    Discrete {
        n: Discrete,
    },

    /// Refer: https://www.gymlibrary.dev/api/spaces/#box
    Box {
        shape: Vec<Discrete>,
        high: Vec<Continous>,
        low: Vec<Continous>,
    },

    // Refer: https://www.gymlibrary.dev/api/spaces/#tuple
    Tuple {
        spaces: Vec<ObsActSpace>,
    },
}

impl ObsActSpace {
    pub fn from_json(info: &Map<String, Value>) -> Self {
        match info["name"].as_str().unwrap() {
            "Discrete" => ObsActSpace::Discrete {
                n: info["n"].as_i64().unwrap() as Discrete,
            },
            "Box" => ObsActSpace::Box {
                shape: as_discrete_item_vec(&info["shape"]),
                high: as_continous_item_vec(&info["high"]),
                low: as_continous_item_vec(&info["low"]),
            },
            "Tuple" => panic!("Parsing for Tuple spaces is not yet implemented"),
            e => panic!("Unrecognized space name: {}", e),
        }
    }

    pub fn items_from_json(&self, vals: &[Value]) -> Vec<ObsActSpaceItem> {
        match self {
            ObsActSpace::Discrete { n: _ } => vals
                .iter()
                .map(|v| ObsActSpaceItem::Discrete(v.as_i64().unwrap() as Discrete))
                .collect::<Vec<_>>(),

            ObsActSpace::Box {
                shape: _,
                high: _,
                low: _,
            } => vals
                .iter()
                .map(|v| ObsActSpaceItem::Continous(v.as_f64().unwrap() as Continous))
                .collect::<Vec<_>>(),

            ObsActSpace::Tuple { spaces: _ } => unimplemented!("Not yet implemented for tuples."),
        }
    }
}

#[derive(Debug)]
pub enum RenderFrame {
    Ansi(String),
    Rgb(Vec<Vec<Vec<u8>>>),
}

impl RenderFrame {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            RenderFrame::Ansi(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_rgb(&self) -> Option<&Vec<Vec<Vec<u8>>>> {
        match self {
            RenderFrame::Rgb(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Transition {
    pub next_state: Discrete,
    pub probability: Continous,
    pub reward: Continous,
    pub done: bool,
}

pub type Transitions = HashMap<(Discrete, Discrete), Vec<Transition>>;

#[derive(Debug)]
pub struct Environment {
    client: GymClient,
    name: String,
    instance_id: String,
    obs_space: ObsActSpace,
    act_space: ObsActSpace,
}

#[derive(Debug)]
pub struct StepInfo {
    pub observation: Vec<ObsActSpaceItem>,
    pub reward: f64,
    pub truncated: bool,
    pub terminated: bool,
    pub info: Value,
}

impl Environment {
    pub fn new(
        client: GymClient,
        name: &str,
        instance_id: &str,
        obs_space: ObsActSpace,
        act_space: ObsActSpace,
    ) -> Self {
        Self {
            client,
            name: name.to_string(),
            instance_id: instance_id.to_string(),
            obs_space,
            act_space,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// The Space object corresponding to valid actions, all valid actions should be contained with the space.
    /// For example, if the action space is of type Discrete and gives the value Discrete(2), this means there
    /// are two valid discrete actions: 0 & 1.
    pub fn action_space(&self) -> &ObsActSpace {
        &self.act_space
    }

    pub fn action_space_sample(&self) -> Vec<ObsActSpaceItem> {
        let url = format!(
            "{}{}/action_space/sample",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_get(&url);
        self.act_space
            .items_from_json(&[obj["action"].as_array().unwrap()[0].clone()])
    }

    /// The Space object corresponding to valid observations, all valid observations should be contained with
    /// the space. For example, if the observation space is of type Box and the shape of the object is (4,),
    /// this denotes a valid observation will be an array of 4 numbers. We can check the box bounds as well with attributes.
    pub fn observation_space(&self) -> &ObsActSpace {
        &self.obs_space
    }

    pub fn reset(&self, seed: Option<usize>) -> Vec<ObsActSpaceItem> {
        let mut body = HashMap::from([]);
        if let Some(seed) = seed {
            let _ = body.insert("seed", seed.to_string());
        }

        let url = format!(
            "{}{}/reset/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_post(&url, &body);
        let obs = obj["observation"].as_array().unwrap();
        self.obs_space.items_from_json(obs)
    }

    pub fn render(&self) -> RenderFrame {
        let url = format!(
            "{}{}/render/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_get(&url);
        let rf = &obj["render_frame"];
        if rf.is_string() {
            return RenderFrame::Ansi(rf.as_str().unwrap().to_string());
        } else if rf.is_array() {
            let arr = rf.as_array().unwrap();
            let mut vec = Vec::with_capacity(arr.len());
            for (i, a) in arr.iter().enumerate() {
                let arr = a.as_array().unwrap();
                vec.push(Vec::with_capacity(arr.len()));
                for (j, a) in arr.iter().enumerate() {
                    let arr = a.as_array().unwrap();
                    vec[i].push(Vec::with_capacity(arr.len()));
                    for a in arr {
                        vec[i][j].push(a.as_i64().unwrap() as u8);
                    }
                }
            }
            RenderFrame::Rgb(vec)
        } else {
            unimplemented!()
        }
    }

    pub fn step(&self, action: &[ObsActSpaceItem]) -> StepInfo {
        let mut req = HashMap::from([]);

        match self.act_space {
            ObsActSpace::Discrete { .. } => {
                if action.len() != 1 {
                    panic!("For Discrete space: Expected only one action.")
                }
                if let ObsActSpaceItem::Discrete(action) = action[0] {
                    req.insert("action", to_value(action).unwrap());
                } else {
                    panic!("For Discrete space: Expected only one action of type Discrete.")
                }
            }

            ObsActSpace::Box { ref shape, .. } => {
                if action.len() != shape[0] as usize {
                    panic!("For Box space: Expected same number of actions as shape.")
                }
                let action: Vec<_> = action
                    .iter()
                    .map(|a| {
                        if let ObsActSpaceItem::Continous(a) = a {
                            *a
                        } else {
                            panic!("For Box space: Actions should all be f64")
                        }
                    })
                    .collect();
                req.insert("action", to_value(action).unwrap());
            }

            // TODO: This space thing is a bad design.
            ObsActSpace::Tuple { .. } => panic!("Actions for Tuple spaces not implemented yet"),
        }
        let url = format!(
            "{}{}/step/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_post(&url, &req);
        let observation = obj["observation"].as_array().unwrap();
        let observation = self.obs_space.items_from_json(observation);

        StepInfo {
            observation,
            reward: obj["reward"].as_f64().unwrap(),
            truncated: obj["truncated"].as_bool().unwrap(),
            terminated: obj["terminated"].as_bool().unwrap(),
            info: obj["info"].clone(),
        }
    }

    pub fn transitions(&self) -> Transitions {
        let url = format!(
            "{}{}/transitions/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_get(&url);
        let obj = obj["transitions"].as_object().unwrap();

        let mut transitions: Transitions = HashMap::new();
        if let (ObsActSpace::Discrete { n: n_s }, ObsActSpace::Discrete { n: n_a }) =
            (self.observation_space(), self.action_space())
        {
            for s in 0..*n_s {
                let s_trans = obj[&s.to_string()].as_object().unwrap();
                for a in 0..*n_a {
                    let a_trans = s_trans[&a.to_string()].as_array().unwrap();
                    let ts = a_trans
                        .iter()
                        .map(|t| {
                            let t = t.as_array().unwrap();
                            Transition {
                                probability: t[0].as_f64().unwrap() as Continous,
                                next_state: t[1].as_i64().unwrap() as Discrete,
                                reward: t[2].as_f64().unwrap() as Continous,
                                done: t[3].as_bool().unwrap(),
                            }
                        })
                        .collect();

                    transitions.insert((s, a), ts);
                }
            }
        } else {
            panic!("Cannot get transition probabilities for environments that dont have discrete observation and action spaces.")
        }

        transitions
    }
}

impl GymClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            base_uri: format!("{host}:{port}"),
            client: Client::builder().build().unwrap(),
        }
    }

    pub fn get_envs(&self) -> HashMap<String, String> {
        let url = self.construct_req_url("/v1/envs/");
        let val = self.http_get(&url);

        let obj = val["all_envs"]
            .as_object()
            .ok_or("No all_envs returned.")
            .unwrap();
        let ret: HashMap<_, _> = obj
            .into_iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();

        ret
    }

    pub fn make_env(
        self,
        env_id: &str,
        max_episode_steps: Option<Discrete>,
        auto_reset: Option<bool>,
        disable_env_checker: Option<bool>,
        kwargs: HashMap<&str, Value>,
    ) -> Environment {
        let mut body = HashMap::<&str, Value>::from([("env_id", to_value(env_id).unwrap())]);
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
        body.insert("kwargs", to_value(kwargs).unwrap());

        let base_url = self.construct_req_url("/v1/envs/");
        let obj = self.http_post(&base_url, &body);
        let inst_id = obj["instance_id"].as_str().unwrap();

        let url = format!("{base_url}{inst_id}/observation_space/");
        let obj = self.http_get(&url);
        let info = obj["info"].as_object().unwrap();
        let obs_space = ObsActSpace::from_json(info);

        let url = format!("{base_url}{inst_id}/action_space/");
        let obj = self.http_get(&url);
        let info = obj["info"].as_object().unwrap();
        let act_space = ObsActSpace::from_json(info);

        Environment::new(self, env_id, inst_id, obs_space, act_space)
    }

    pub fn construct_req_url(&self, path: &str) -> String {
        format!("{}{}", self.base_uri, path)
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

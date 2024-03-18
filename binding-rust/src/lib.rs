extern crate rand;
extern crate reqwest;
extern crate serde;
extern crate serde_json;

// TODO: DRY

use rand::Rng;
use reqwest::{
    blocking::*,
    header::{HeaderMap, HeaderValue, CONTENT_TYPE},
};
use serde::ser::Serialize;
use serde_json::{from_value, to_value, Map, Value};
use std::collections::HashMap;
use std::error::Error;
use std::result::Result;

#[derive(Debug)]
pub struct GymClient {
    base_uri: String,
    client: Client,
}

pub type GymResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
pub enum Space {
    Discrete {
        n: u64,
    },

    Box {
        shape: Vec<u64>,
        high: Vec<f64>,
        low: Vec<f64>,
    },

    Tuple {
        spaces: Vec<Box<Space>>,
    },
}

pub enum SampleItem {
    U64(u64),
    F64(f64),
}

impl Space {
    pub fn from_json(info: &Map<String, Value>) -> GymResult<Self> {
        match info["name"].as_str().ok_or("No name returned.")? {
            "Discrete" => {
                let n = GymClient::value_to_number::<u64>(&info["n"]);
                Ok(Space::Discrete { n })
            }
            "Box" => {
                let shape = GymClient::value_to_vec::<u64>(&info["shape"]);
                let high = GymClient::value_to_vec::<f64>(&info["high"]);
                let low = GymClient::value_to_vec::<f64>(&info["low"]);
                Ok(Space::Box { shape, high, low })
            }
            "Tuple" => panic!("Parsing for Tuple spaces is not yet implemented"),
            e => panic!("Unrecognized space name: {}", e),
        }
    }

    pub fn sample(&self) -> Vec<SampleItem> {
        let mut rng = rand::thread_rng();
        match self {
            Space::Discrete { n } => {
                vec![SampleItem::U64(rng.gen::<u64>() % n)]
            }
            Space::Box {
                ref shape,
                ref high,
                ref low,
            } => {
                let mut ret = Vec::with_capacity(shape.iter().map(|x| *x as usize).product());
                let mut index = 0;
                for &i in shape {
                    for _ in 0..i {
                        ret.push(SampleItem::F64(rng.gen_range(low[index]..high[index])));
                        index += 1;
                    }
                }
                ret
            }
            Space::Tuple { ref spaces } => {
                let mut ret = Vec::new();
                for space in spaces {
                    ret.extend(space.sample());
                }
                ret
            }
        }
    }
}

#[derive(Debug)]
pub struct Transtion {
    pub next_state: u64,
    pub probability: f64,
    pub reward: f64,
    pub done: bool,
}

#[derive(Debug)]
pub struct Environment {
    client: GymClient,
    instance_id: String,
    obs_space: Space,
    act_space: Space,
}

#[derive(Debug)]
pub struct State {
    pub observation: Vec<f64>,
    pub reward: f64,
    pub truncated: bool,
    pub terminated: bool,
    pub info: Value,
}

impl Environment {
    pub fn new(client: GymClient, instance_id: &str, obs_space: Space, act_space: Space) -> Self {
        Self {
            client,
            instance_id: instance_id.to_string(),
            obs_space,
            act_space,
        }
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// The Space object corresponding to valid actions, all valid actions should be contained with the space.
    /// For example, if the action space is of type Discrete and gives the value Discrete(2), this means there
    /// are two valid discrete actions: 0 & 1.
    pub fn action_space(&self) -> &Space {
        &self.act_space
    }

    /// The Space object corresponding to valid observations, all valid observations should be contained with
    /// the space. For example, if the observation space is of type Box and the shape of the object is (4,),
    /// this denotes a valid observation will be an array of 4 numbers. We can check the box bounds as well with attributes.
    pub fn observation_space(&self) -> &Space {
        &self.obs_space
    }

    pub fn reset(&self, seed: Option<usize>) -> GymResult<Vec<f64>> {
        let mut body = HashMap::from([]);
        if let Some(seed) = seed {
            let _ = body.insert("seed", seed.to_string());
        }

        let url = format!(
            "{}{}/reset/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_post(&url, &body)?;
        let obs = obj["observation"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>();
        Ok(obs)
    }

    pub fn render(&self) -> GymResult<String> {
        let url = format!(
            "{}{}/render/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_get(&url)?;
        let render_frame = obj["render_frame"].as_str().unwrap();
        let render_frame = render_frame.replace("\\u", "\\x").replace(['{', '}'], "");
        Ok(render_frame)
    }

    pub fn step(&self, action: Vec<SampleItem>) -> GymResult<State> {
        let mut req = HashMap::from([]);

        match self.act_space {
            Space::Discrete { .. } => {
                if action.len() != 1 {
                    panic!("For Discrete space: Expected only one action.")
                }
                if let SampleItem::U64(action) = action[0] {
                    req.insert("action", to_value(action).unwrap());
                } else {
                    panic!("For Discrete space: Expected only one action of type u64.")
                }
            }

            Space::Box { ref shape, .. } => {
                if action.len() != shape[0] as usize {
                    panic!("For Box space: Expected same number of actions as shape.")
                }
                let action: Vec<f64> = action
                    .iter()
                    .map(|a| {
                        if let SampleItem::F64(a) = a {
                            *a
                        } else {
                            panic!("For Box space: Actions should all be f64")
                        }
                    })
                    .collect();
                req.insert("action", to_value(action).unwrap());
            }

            // TODO: This space thing is a bad design.
            Space::Tuple { .. } => panic!("Actions for Tuple spaces not implemented yet"),
        }
        let url = format!(
            "{}{}/step/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_post(&url, &req)?;

        Ok(State {
            observation: from_value(obj["observation"].clone()).unwrap(),
            reward: obj["reward"].as_f64().unwrap(),
            truncated: obj["truncated"].as_bool().unwrap(),
            terminated: obj["terminated"].as_bool().unwrap(),
            info: obj["info"].clone(),
        })
    }

    pub fn get_transitions(&self, state: u64, action: u64) -> GymResult<Vec<Transtion>> {
        let url = format!(
            "{}{}/transitions/{state}/{action}/",
            self.client.construct_req_url("/v1/envs/"),
            self.instance_id
        );
        let obj = self.client.http_get(&url)?;
        let transtions = obj["transitions"].as_array().unwrap();
        let transtions = transtions
            .iter()
            .map(|t| Transtion {
                next_state: GymClient::value_to_number::<u64>(&t["next_state"]),
                probability: GymClient::value_to_number::<f64>(&t["p"]),
                reward: GymClient::value_to_number::<f64>(&t["reward"]),
                done: t["done"].as_bool().unwrap(),
            })
            .collect();

        Ok(transtions)
    }
}

impl GymClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            base_uri: format!("{host}:{port}"),
            client: Client::builder().build().unwrap(),
        }
    }

    pub fn get_envs(&self) -> Result<HashMap<String, String>, Box<dyn Error>> {
        let url = self.construct_req_url("/v1/envs/");
        let val = self.http_get(&url)?;

        let obj = val["all_envs"].as_object().ok_or("No all_envs returned.")?;
        let ret: HashMap<_, _> = obj
            .into_iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();

        Ok(ret)
    }

    pub fn make_env(self, env_id: &str, render_mode: Option<&str>) -> GymResult<Environment> {
        let mut body = HashMap::from([("env_id", env_id.to_string())]);
        if let Some(render_mode) = render_mode {
            body.insert("render_mode", render_mode.to_string());
        }

        let base_url = self.construct_req_url("/v1/envs/");
        let obj = self.http_post(&base_url, &body)?;
        let inst_id = obj["instance_id"].as_str().unwrap();

        let url = format!("{base_url}{inst_id}/observation_space/");
        let obj = self.http_get(&url)?;
        let info = obj["info"].as_object().unwrap();
        let obs_space = Space::from_json(info);

        let url = format!("{base_url}{inst_id}/action_space/");
        let obj = self.http_get(&url)?;
        let info = obj["info"].as_object().unwrap();
        let act_space = Space::from_json(info);

        Ok(Environment::new(self, inst_id, obs_space?, act_space?))
    }

    pub fn construct_req_url(&self, path: &str) -> String {
        format!("{}{}", self.base_uri, path)
    }

    fn value_to_number<T>(val: &Value) -> T
    where
        T: std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Debug,
    {
        val.as_str().unwrap().parse::<T>().unwrap()
    }

    fn value_to_vec<T>(val: &Value) -> Vec<T>
    where
        T: std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Debug,
    {
        val.as_array()
            .unwrap()
            .iter()
            .map(Self::value_to_number::<T>)
            .collect::<Vec<_>>()
    }

    fn http_get(&self, url: &str) -> GymResult<Value> {
        let res = self
            .client
            .get(url)
            .headers(Self::construct_common_headers())
            .send();
        let ret = res?.json::<Value>()?;

        Ok(ret)
    }

    fn http_post<T: Serialize>(&self, url: &str, body: &HashMap<&str, T>) -> GymResult<Value> {
        let res = self
            .client
            .post(url)
            .headers(Self::construct_common_headers())
            .json(body)
            .send();
        let ret = res?.json::<Value>()?;

        Ok(ret)
    }

    fn construct_common_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_space_sample() {
        let n = 15;
        let discrete_space = Space::Discrete { n };
        for _ in 0..10 {
            let sample = discrete_space.sample();
            assert!(sample.len() == 1);
            if let SampleItem::U64(sample) = sample[0] {
                assert!(sample < n);
            } else {
                panic!("For discrete sample space samples must SampleItem::U64.");
            }
        }
    }

    #[test]
    fn test_box_space_sample() {
        let shape = vec![5];
        let high = vec![1., 2., 3., 4., 5.];
        let low = vec![-1., -2., -3., -4., -5.];
        let box_space = Space::Box {
            shape: shape.clone(),
            high: high.clone(),
            low: low.clone(),
        };
        for _ in 0..10 {
            let sample = box_space.sample();
            assert_eq!(sample.len(), shape[0] as usize);
            for i in 0..5 {
                if let SampleItem::F64(sample) = sample[i] {
                    assert!(low[i] <= sample && sample <= high[i]);
                } else {
                    panic!("For discrete sample space samples must SampleItem::F64.");
                }
            }
        }
    }

    #[test]
    fn test_tuple_space_sample() {
        let n = 15;
        let discrete_space = Space::Discrete { n };
        let shape = vec![5];
        let high = vec![1., 2., 3., 4., 5.];
        let low = vec![-1., -2., -3., -4., -5.];
        let box_space = Space::Box {
            shape: shape.clone(),
            high: high.clone(),
            low: low.clone(),
        };

        let tuple_space = Space::Tuple {
            spaces: vec![Box::new(discrete_space), Box::new(box_space)],
        };
        for _ in 0..10 {
            let sample = tuple_space.sample();
            assert_eq!(sample.len(), (shape[0] + 1) as usize);

            if let SampleItem::U64(sample) = sample[0] {
                assert!(sample < n);
            } else {
                panic!("For discrete sample space samples must SampleItem::U64.");
            }

            for i in 1..6 {
                if let SampleItem::F64(sample) = sample[i] {
                    assert!(low[i - 1] <= sample && sample <= high[i - 1]);
                } else {
                    panic!("For discrete sample space samples must SampleItem::F64.");
                }
            }
        }
    }
}

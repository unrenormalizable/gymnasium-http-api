use crate::*;
use base64::prelude::*;
use flate2::read::ZlibDecoder;
use std::io::prelude::*;
use std::mem::size_of;

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

fn map_py_type_name_to_rust(name: &str) -> &str {
    match name {
        "int32" => "i32",
        "int64" => "i64",
        "float32" => "f32",
        "float64" => "f64",
        _ => "unknown",
    }
}

use pyo3::prelude::*;
use std::collections::HashMap;

struct TrieNode {
    children: HashMap<char, TrieNode>,
    token_id: Option<usize>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            token_id: None,
        }
    }
}

#[pyclass]
pub struct Trie {
    root: TrieNode,
    next_id: usize, // for assigning unique IDs to tokens
    unk_token_set: bool,
    unk_token_id: usize,
}

// A Trie: See https://en.wikipedia.org/wiki/Trie
// This is a data structure that allows for tokenizing a stream of text
// such that the longest possible tokens are recognized first.
//
// To explain how this works, let's first consider how we add new tokens.
// Let's say we have four possible tokens: 'A', 'B', 'AA', 'AB'
// The trie always has an empty root node. There will be two children:
// the node 'A' with token_id 0 and 'B' with token_id 1. The node 'B' has
// no children since we have no tokens that start with 'B' and continue to
// another character.
// The node 'A' has two children, one other node with 'A' with token_id of 2
// and a node 'B' with token_id of 3.
// Now, to tokenize the following string: 'ABAABA'
// We start from the beginning of the string, continuing until we no longer have
// the substring in our tokens. The first character is 'A'; going down the trie
// we have a node that starts with 'A'. The next character is 'B', and our 'A' node
// has a child that starts with 'B'. The character after is 'A', but our last node,
// with token_id = 3, has no children, so we have our first token 'AB' with token_id 3.
// Similarly, we have 'AA' token_id 2, 'B' with token_id 1, and 'A' with token_id 0.
// So, 'ABAABA' -> [3, 2, 1, 0]
#[pymethods]
impl Trie {
    #[new]
    pub fn new(unk_token_id: Option<usize>) -> Self {
        Trie {
            root: TrieNode::new(),
            next_id: 0, // We start the IDs at 0
            unk_token_set: unk_token_id.is_some(),
            unk_token_id: unk_token_id.unwrap_or(0),
        }
    }

    // Function responsible for figuring out the tree structure
    // Children are represented as dictionaries to make the search simpler.
    // In fact, for our purposes where the number of children will be small,
    // it is probably faster to use lists.
    pub fn add(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        if node.token_id.is_none() {
            node.token_id = Some(self.next_id);
            self.next_id += 1;
            if !self.unk_token_set {
                self.unk_token_id = self.next_id;
            }
        }
    }

    // Tokenizing function. Does what is described in the comment above.
    // You can see how we keep going through the characters until we hit a node
    // that has no children.
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![];
        let mut start = 0;
        
        while start < text.len() {
            let mut node = &self.root;
            let mut matched = false;
            let mut end = start;
            for ch in text[start..].chars() {
                if let Some(next_node) = node.children.get(&ch) {
                    // If the character matches a child, we go to the next node
                    node = next_node;
                    end += ch.len_utf8();
                    if node.token_id.is_some() {  // If at the leaf, we have our token
                        matched = true;
                        break;
                    }
                } else {  // This means we never matched, so it is an '<unk>' token
                    break;
                }
            }
            
            if matched {
                tokens.push(node.token_id.unwrap());
                start = end;
            } else {
                tokens.push(self.unk_token_id); // Assign unknown token ID
                start += text[start..].chars().next().unwrap().len_utf8();
            }
        }

        tokens
    }
}

#[pymodule]
fn rust_trie(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Trie>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn test_trie() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let trie_module = PyModule::new(py, "trie_module").unwrap();
        let locals = [("trie", trie_module)].into_py_dict(py);
        let py_trie: PyObject = py
            .eval("trie.Trie()", Some(locals), None)
            .unwrap()
            .into();
        py_trie.call_method0("add", "[CLS]").unwrap();
        let tokens: Vec<usize> = py_trie
            .call_method1("tokenize", ("[CLS] This is a test",))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(tokens, vec![0, 1, 1, 1, 1]);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn test_trie() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let trie_mod = PyModule::new(py, "trie_module").unwrap();
        let locals = [("trie", trie_mod)].into_py_dict(py);
        let py_trie: PyObject = py
            .eval("trie.Trie()", Some(locals), None)
            .unwrap()
            .into();
        py_trie.call_method0("add", "<cls>").unwrap();
        let tokens: Vec<usize> = py_trie
            .call_method1("tokenized", ("<cls> This is a test",))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(tokens, vec![0, 1, 1, 1, 1]);
    }
}


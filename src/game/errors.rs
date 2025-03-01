use std::fmt;

#[derive(Debug, Clone)]
pub struct InvalidAction(String);

impl InvalidAction {
    pub fn new(msg: String) -> Self {
        InvalidAction(msg)
    }
}

impl fmt::Display for InvalidAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid action: {}", self.0)
    }
}

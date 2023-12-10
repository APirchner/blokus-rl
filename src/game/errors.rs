use std::fmt;

#[derive(Debug, Clone)]
pub struct InvalidAction;

impl fmt::Display for InvalidAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid action")
    }
}

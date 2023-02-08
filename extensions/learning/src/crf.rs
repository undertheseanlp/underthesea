// create struct Student
struct Student {
    name: String,
    age: u8,
    grade: u8,
    // add classes
    classes: Vec<Class>
}

// create struct Class
struct Class {
    name: String,
}

// implement method add_class for Student
impl Student {
    fn add_class(&mut self, class: Class){
        self.classes.push(class);
    }
}

// create struct CRF
struct CRF {
}

// implement method train for CRF
impl CRF {
    fn train(&mut self){
        // do something
    }

    // implement method tag with parameter features
    fn tag(&mut self, features: Vec<String>) -> Vec<String> {
        vec![]
    }
}
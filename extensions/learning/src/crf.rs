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
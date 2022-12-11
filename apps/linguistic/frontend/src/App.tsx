import React from 'react';
import logo from './logo.svg';
import './App.css';
import { Button, Alert, Breadcrumb } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
// import { Menu } from './components/Menu';

function App() {
  return (
    
    <div className="App">
      {/* <Menu></Menu> */}
      <header className="App-header">
        <Alert variant="warning">This is an alert</Alert> 
        <Button>Test Button</Button>
      </header>
    </div>
  );
}

export default App;

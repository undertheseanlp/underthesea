import Container from 'react-bootstrap/Container'
import Navbar from 'react-bootstrap/Navbar'
import Form from 'react-bootstrap/Form'

export function NavbarMain() {
  return (
    <Navbar bg="light" expand="lg">
      <Container>
        <Navbar.Brand href="#home">ToMo</Navbar.Brand>
        <Form>
          <Form.Control
            type="text"
            id="search"
          />
        </Form>
      </Container>
    </Navbar>
  )
}

import 'bootstrap/dist/css/bootstrap.min.css'
import { Route, Switch, BrowserRouter } from 'react-router-dom'
import './App.css'
// import NavbarMain from './components/NavbarMain'
import AppbarMain from './components/AppbarMain'
import MainDrawer from './components/MainDrawer'
import { ArticleDetail } from './features/articles/ArticleDetail'
import { ArticleList } from './features/articles/ArticleList'
import { Counter } from './features/counter/Counter'
import Demo from './features/demo/Demo'
import Box from '@mui/material/Box'
import CssBaseline from '@mui/material/CssBaseline'

function App() {
  return (
    <div>
      <BrowserRouter>
        <Switch>
          <Route path="/" exact>
            <Box sx={{ display: 'flex' }}>
              <CssBaseline />
              <AppbarMain></AppbarMain>
              <MainDrawer></MainDrawer>
              <ArticleList />
            </Box>
          </Route>
          <Route path="/ArticleList">
            <ArticleList />
          </Route>
          <Route path="/Demo">
            <Demo />
          </Route>
          <Route path="/Counter">
            <Counter />
          </Route>
          <Route path="/ArticleDetail/:id">
            <ArticleDetail />
          </Route>
        </Switch>
      </BrowserRouter>
    </div>
  )
}

export default App

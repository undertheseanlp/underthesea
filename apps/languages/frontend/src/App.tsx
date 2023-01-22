import Box from '@mui/material/Box'
import CssBaseline from '@mui/material/CssBaseline'
import 'bootstrap/dist/css/bootstrap.min.css'
import { BrowserRouter, Route, Switch } from 'react-router-dom'
import './App.css'
import AppbarMain from './components/AppbarMain'
import MainDrawer from './components/MainDrawer'
import { Utilities } from './features/utilities/Utilities'
import Dictionary from './features/utilities/Dictionary'
import WordNew from './features/utilities/WordNew'
import { ArticleDetail } from './features/articles/ArticleDetail'
import { ArticleList } from './features/articles/ArticleList'
import ArticleNew from './features/articles/ArticleNew'
import { Counter } from './features/counter/Counter'
import Demo from './features/demo/Demo'

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
            <Box sx={{ display: 'flex' }}>
              <CssBaseline />
              <AppbarMain></AppbarMain>
              <MainDrawer></MainDrawer>
              <ArticleList />
            </Box>
          </Route>
          <Route path="/Utilities" exact>
            <Box sx={{ display: 'flex' }}>
              <CssBaseline />
              <AppbarMain></AppbarMain>
              <MainDrawer></MainDrawer>
              <Utilities />
            </Box>
          </Route>
          <Route path="/Demo">
            <Demo />
          </Route>
          <Route path="/Dictionary">
            <Dictionary />
          </Route>
          <Route path="/WordNew">
            <WordNew />
          </Route>
          <Route path="/Counter">
            <Counter />
          </Route>
          <Route path="/ArticleDetail/:id">
            <ArticleDetail />
          </Route>
          <Route path="/ArticleNew">
            <ArticleNew />
          </Route>
        </Switch>
      </BrowserRouter>
    </div>
  )
}

export default App

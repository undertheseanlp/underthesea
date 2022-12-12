import 'bootstrap/dist/css/bootstrap.min.css'
import { Route, Switch } from 'react-router-dom'
import './App.css'
import { NavbarMain } from './components/NavbarMain'
import { ArticleDetail } from './features/articles/ArticleDetail'
import { ArticleList } from './features/articles/ArticleList'
import { Counter } from './features/counter/Counter'

function App() {
  return (
    <div>
      <NavbarMain></NavbarMain>
      <Switch>
        <Route path="/" exact>
          <ArticleList />
        </Route>
        <Route path="/ArticleList">
          <ArticleList />
        </Route>
        <Route path="/Counter">
          <Counter />
        </Route>
        <Route path="/ArticleDetail/:id">
          <ArticleDetail />
        </Route>
      </Switch>
    </div>
  )
}

export default App

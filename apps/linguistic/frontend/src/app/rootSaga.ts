import { all } from 'redux-saga/effects'
import ArticlesSaga from '../features/articles/ArticlesSaga';
import counterSaga from '../features/counter/counterSaga'

export default function* rootSaga(){
  console.log("rootSaga");
  yield all([
    counterSaga(),
    ArticlesSaga()
  ])
}
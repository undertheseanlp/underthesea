import { all } from 'redux-saga/effects'
import counterSaga from '../features/counter/counterSaga'

function* helloSaga(){
  console.log("hello");
}

export default function* rootSaga(){
  console.log("rootSaga");
  yield all([
    helloSaga(),
    counterSaga()
  ])
}
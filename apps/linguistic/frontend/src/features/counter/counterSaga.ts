import { delay, takeEvery, put, takeLatest } from '@redux-saga/core/effects'
import { PayloadAction } from '@reduxjs/toolkit'
import { increment, incrementSaga, incrementSagaSuccess } from './counterSlice';

function* handleIncrementSaga(action: PayloadAction<number>){
  console.log('waiting 1 seconds');
  // wait 1 seconds
  yield delay(1000);

  console.log('Done, dispatch action');

  // dispatch action success
  yield put(incrementSagaSuccess(action.payload));
}

export default function* counterSaga(){
  console.log("counterSaga");
  yield takeEvery(incrementSaga.toString(), handleIncrementSaga)
  // yield takeLatest(incrementSaga.toString(), handleIncrementSaga)
}
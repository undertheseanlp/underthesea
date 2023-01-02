import { call, put, takeLatest } from "redux-saga/effects";
import ArticleApi from "../../api/articleApi";
import { Article, ListResponse } from "../../models";
import { ArticlesActions } from "./ArticlesSlice";

function* getAll(){
  try {
    console.log('saga function* getAll');
    const response: ListResponse<Article> = yield call(ArticleApi.getAll);
    console.log(response);
    yield put(ArticlesActions.getAllSuccess(response));
  } catch (error) {
    console.log(error);
    yield put(ArticlesActions.getAllFailed);
  }
}

export default function* ArticlesSaga() {
  // watch getAll 
  console.log("Articles Saga");
  yield takeLatest(ArticlesActions.getAll, getAll);
}
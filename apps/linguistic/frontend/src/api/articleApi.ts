import { Article, ListResponse } from "../models";
import axiosClient from "./client";

const ArticleApi = {
  getAll(): Promise<ListResponse<Article>> {
    console.log('ArticleApi.getAll');
    const url = '/articles/';
    return axiosClient.get(url, { params: {limit: 20, offset: 0}});
  }
}

export default ArticleApi;
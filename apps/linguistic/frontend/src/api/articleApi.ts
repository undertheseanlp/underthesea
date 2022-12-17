import { Article, ListResponse } from "../models";
import axiosClient from "./client";

const articleApi = {
  getAll(): Promise<ListResponse<Article>> {
    const url = '/articles/';
    return axiosClient.get(url, { params: {limit: 10, offset: 0}});
  }
}

export default articleApi;
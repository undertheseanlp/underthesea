import { Collection, ListResponse } from "../models";
import axiosClient from "./client";

const collectionApi = {
  getAll(): Promise<ListResponse<Collection>> {
    const url = '/collections/';
    return axiosClient.get(url, { params: {limit: 10, offset: 0}});
  }
}

export default collectionApi;
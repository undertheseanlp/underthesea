import { Collection, ListResponse } from "../models";
import axiosClient from "./client";

const collectionApi = {
  getAll(): Promise<ListResponse<Collection>> {
    const url = '/collections/';
    return axiosClient.get(url, { params: {limit: 10, offset: 0}});
  },

  getById(id: string): Promise<Collection> {
    const url = `/collections/{id}`;
    return axiosClient.get(url);
  },

  add(data: Collection): Promise<Collection> {
    const url = '/collections/';
    return axiosClient.post(url, data);
  },

  update(data: Collection): Promise<Collection> {
    const url = '/collections/';
    return axiosClient.patch(url, data);
  },

  remove(id: string): Promise<any> {
    const url = `/collections/${id}`;
    return axiosClient.delete(url);
  },
}

export default collectionApi;
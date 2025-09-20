import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { RootState } from "../../app/store";
import { Article, FilterParams, ListResponse, PaginationParams } from "../../models";

export interface ArticlesState {
  data: Article[];
  isLoading: boolean;
  filter?: FilterParams;
  pagination?: PaginationParams;
}

const initialState: ArticlesState = {
  data: [],
  isLoading: false,
  pagination: {
    limit: 100,
    offset: 0
  }
};

export const ArticlesSlice = createSlice({
  name: 'Articles',
  initialState,
  reducers: {
    getAll: (state) => {
      state.isLoading = true;
    },
    getAllSuccess: (state, action: PayloadAction<ListResponse<Article>>) => {
      state.data = action.payload.results;
      state.isLoading = false;
    },
    getAllFailed: (state) => {
      state.isLoading = false;
    }
  }
})

// Actions
export const ArticlesActions = ArticlesSlice.actions;

// Selectors
export const selectArticles = (state: RootState) => state.articles.data;
export const selectArticlesIsLoading = (state: RootState) => state.articles.isLoading;
export const selectArticlesFilter = (state: RootState) => state.articles.filter;
export const selectArticlesPagination = (state: RootState) => state.articles.pagination;

// Reducer
const ArticlesReducer = ArticlesSlice.reducer;

export default ArticlesReducer;
export interface ListResponse<T> {
  results: T[],
  count: number;
  next: string | null;
  previous: string | null;
}

export interface FilterParams {
  limit: number;
  offset: number;
}

export interface PaginationParams {
  limit: number;
  offset: number;
}
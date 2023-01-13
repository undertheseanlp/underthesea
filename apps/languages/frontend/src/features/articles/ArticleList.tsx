import Box from '@mui/material/Box'
import Toolbar from '@mui/material/Toolbar'
import { useEffect } from 'react'
import Card from 'react-bootstrap/Card'
import { Link } from 'react-router-dom'
import Pagination from '@mui/material/Pagination';
import { useAppDispatch, useAppSelector } from '../../app/hooks'
import { ArticlesActions, selectArticles } from './ArticlesSlice'

export function ArticleList() {
  const linkStyle = {
    textDecoration: 'none',
    color: 'black',
  }
  const dispatch = useAppDispatch()
  const articles = useAppSelector(selectArticles)
  console.log(articles)

  useEffect(() => {
    dispatch(ArticlesActions.getAll())
    console.log('useEffect')
  }, [dispatch])

  const changePageSize = (params: any, page: any) => {
    console.log('changePageSize', params, page);
  }


  return (
    <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
      <Toolbar />
      {articles.map((article, index) => (
        <Box mt={1}>
          <Link to={`ArticleDetail/` + article.id} style={linkStyle}>
            <Card>
              <Card.Body>
                <Card.Title>{article.title}</Card.Title>
                <Card.Text>{article.description}</Card.Text>
              </Card.Body>
            </Card>
          </Link>
        </Box>
      ))}
      <br/>
      <Pagination count={11} defaultPage={1} siblingCount={0} onChange={changePageSize}  />
    </Box>
  )
}

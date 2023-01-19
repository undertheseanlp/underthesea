import Box from '@mui/material/Box'
import Toolbar from '@mui/material/Toolbar'
import Card from '@mui/material/Card'
import CardActions from '@mui/material/CardActions'
import CardContent from '@mui/material/CardContent'
import Button from '@mui/material/Button'
import Typography from '@mui/material/Typography'
import { Link } from 'react-router-dom'

export function Utilities() {
  return (
    <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
      <Toolbar />
      <h3>Utilities</h3>
      <hr />

      <Card sx={{ minWidth: 275 }}>
        <CardContent>
          <Typography variant="h5" color="text.secondary">
            Dictionary
          </Typography>
          <Typography variant="body1">
            A comprehensive resource for language and meaning
            <br />
          </Typography>
        </CardContent>
        <CardActions>
          <Link to="Dictionary">
            <Button size="medium">Organize</Button>
          </Link>
        </CardActions>
      </Card>
    </Box>
  )
}

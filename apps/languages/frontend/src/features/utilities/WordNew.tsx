import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import { Box, Typography } from '@mui/material'
import AppBar from '@mui/material/AppBar'
import IconButton from '@mui/material/IconButton'
import { styled } from '@mui/material/styles'
import Toolbar from '@mui/material/Toolbar'
import { Link } from 'react-router-dom'
import { Article } from '../../models'
import TextField from '@mui/material/TextField'
import Button from '@mui/material/Button'
import FormControl from '@mui/material/FormControl'

const StyledToolbar = styled(Toolbar)(({ theme }) => ({
  alignItems: 'flex-start',
  paddingTop: theme.spacing(1),
  paddingBottom: theme.spacing(2),
  // Override mediam queries injected by theme.mixins.toolbar
  '@media all': {
    minHeight: 128,
  },
}))

const handleArticleFormSubmit = (formValues: Article) => {}

function WordNew() {
  return (
    <div>
      <AppBar color="transparent" elevation={0}>
        <StyledToolbar>
          <IconButton
            component={Link}
            to="/Dictionary"
            size="medium"
            aria-label="display more actions"
            edge="end"
            color="inherit"
          >
            <ArrowBackIcon />
          </IconButton>
        </StyledToolbar>
      </AppBar>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        <Typography variant="h4">New Word</Typography>
      </Box>

      <Box sx={{ p: 3 }}>
        <FormControl fullWidth sx={{ m: 1 }}>
          <TextField id="standard-basic" label="Word" variant="standard" />
          <Box mt={12}>
            <Button type="submit" variant="contained" color="primary">
              Save
            </Button>
          </Box>
        </FormControl>
      </Box>
    </div>
  )
}

export default WordNew
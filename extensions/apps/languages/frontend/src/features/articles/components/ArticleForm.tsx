import React from 'react'
import { Article } from '../../../models'
import { useForm } from 'react-hook-form'
import Box from '@mui/material/Box'
import { InputField } from '../../../components/FormFields'
import Button from '@mui/material/Button'

export interface StudentFormProps {
  initialValues?: Article
  onSubmit?: (formValues: Article) => void
}

export default function ArticleForm({
  initialValues,
  onSubmit,
}: StudentFormProps) {
  const { control, handleSubmit } = useForm<Article>({
    defaultValues: initialValues,
  })

  const handleFormSubmit = (formValues: Article) => {
    console.log('Submit', formValues);
  }

  return <Box maxWidth={400}>
    <form onSubmit={handleSubmit(handleFormSubmit)}>
      <InputField name="title" control={control} label="Title"/>
      <InputField name="description" control={control} label="Description"/>
      <InputField name="text" control={control} label="Text"/>
      <Box mt={3}>
        <Button type="submit" variant="contained" color="primary">Save</Button>
      </Box>
    </form>
  </Box>
}

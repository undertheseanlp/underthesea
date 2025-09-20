import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"
import { styled } from '@material-ui/core/styles';
import ReactJson from 'react-json-view'

interface State {
  json_object: object
}

/**
 * This is a React-based component template. The `render()` function is called
 * automatically when your component should be re-rendered.
 */
class JsonViewer extends StreamlitComponentBase<State> {
  constructor(props: any){
    super(props)
    console.log(props)
    const data = this.props.args["json_object"]
    this.state = {
      "json_object": data
    }
    Streamlit.setComponentValue(data)
  }

  private onChanged = (e: any): boolean => {
     var data = e.updated_src
     Streamlit.setComponentValue(data)
     this.setState({json_object: data})
     return true
  }
  public render = (): ReactNode => {
    const vMargin = 7
    const hMargin = 20
    const StyledReactJson = styled(ReactJson)({
      margin: `${vMargin}px ${hMargin}px`,
      width: this.props.width - (hMargin * 2)
    });

    return (
      <StyledReactJson
        src={this.state["json_object"]}
        onEdit={this.onChanged}
        onDelete={this.onChanged}
        onAdd={this.onChanged}
        name={null}
        displayDataTypes={false}
      />
    )
  }

}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
export default withStreamlitConnection(JsonViewer)

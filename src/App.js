import React, { useRef, useEffect, useState } from 'react'
import { makeStyles } from '@material-ui/core/styles'
import CssBaseline from '@material-ui/core/CssBaseline'
import AppBar from '@material-ui/core/AppBar'
import Toolbar from '@material-ui/core/Toolbar'
import Paper from '@material-ui/core/Paper'
import Grid from '@material-ui/core/Grid'
import Typography from '@material-ui/core/Typography'
import Button from '@material-ui/core/Button'
import ButtonGroup from '@material-ui/core/ButtonGroup'
import CanvasDraw from "react-canvas-draw"
import * as tf from '@tensorflow/tfjs'
import LinearProgress from '@material-ui/core/LinearProgress'
import Snackbar from '@material-ui/core/Snackbar'
import Alert from '@material-ui/lab/Alert'
import Box from '@material-ui/core/Box'
import useMediaQuery from '@material-ui/core/useMediaQuery'
import useTheme from '@material-ui/core/styles/useTheme'


const useStyles = makeStyles((theme) => ({
  appBar: {
    position: 'relative',
  },
  layout: {
    width: 'auto',
    marginLeft: theme.spacing(2),
    marginRight: theme.spacing(2),
    [theme.breakpoints.up(600 + theme.spacing(2) * 2)]: {
      width: '80vw',
      marginLeft: 'auto',
      marginRight: 'auto',
    },
  },
  paper: {
    marginTop: theme.spacing(3),
    padding: theme.spacing(2),
    width: 'auto',
    [theme.breakpoints.up(600 + theme.spacing(3) * 2)]: {
      marginTop: theme.spacing(6),
      padding: theme.spacing(3),
    },
  },
  canvasPaper: {
    marginTop: theme.spacing(3),
    marginBottom: theme.spacing(3),
    padding: theme.spacing(2),
    [theme.breakpoints.up(600 + theme.spacing(3) * 2)]: {
      marginTop: theme.spacing(6),
      marginBottom: theme.spacing(6),
      width: '448px',
      padding: theme.spacing(3),
    },
  },
  canvasElement: {
    margin: '0 auto'
  },
  btnGroup: {
    margin: "2em 1em"
  }
}));

function ProbIndicator({ v, i }) {
  return (
    <Box display="block" alignItems="center">
      <Box width="100%" mr={1}>
        <LinearProgress variant="determinate" value={v * 100} />
      </Box>
      <Box minWidth={35}>
        <Typography variant="body2" color="textSecondary">{`${i}: ${Math.round(
          v * 100,
        )}%`}</Typography>
      </Box>
    </Box>
  )
}

export default function App() {
  const theme = useTheme()
  const isSM = useMediaQuery(theme.breakpoints.down('sm'))
  const canvansSize = isSM ? 300 : 400
  const classes = useStyles()
  const canvasRef = useRef(null)
  const [model, setModel] = useState(null)
  const [open, setOpen] = useState(false)
  const [predRes, setPredRes] = useState([])
  async function recognize(model) {
    const image = tf.browser.fromPixels(canvasRef.current.canvasContainer.children[1])
    const img = image
      .toFloat()
      .mean(2)
      .divNoNan(255)
      .step(0)
      .reshape([canvansSize, canvansSize, 1])
      .conv2d(tf.tensor4d([0.094, 0.118, 0.094, 0.118, 0.148, 0.118, 0.094, 0.118, 0.094], [3, 3, 1, 1]), 1, 'same')
      .resizeBilinear([28, 28])
      .reshape([1, 28, 28, 1])
    await tf.browser.toPixels(img.reshape([28, 28]), document.getElementById('qwq'))
    const res = model.predict(img)
    setPredRes((await res.array())[0])
    image.dispose()
    img.dispose()
    res.dispose()
  }
  useEffect(() => {
    async function loadModel() {
      const model = await tf.loadLayersModel('/model.json')
      setModel(model)
      setOpen(true)
      await recognize(model)
    }
    loadModel()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <>
      <CssBaseline />
      <Snackbar open={open} autoHideDuration={3000} onClose={() => { setOpen(false) }}>
        <Alert severity="success" elevation={6} variant="filled">
          Successfully Load Tensorflow Model
        </Alert>
      </Snackbar>
      {!model && <LinearProgress style={{ zIndex: 2000 }} />}
      <AppBar position="absolute" color="default" className={classes.appBar}>
        <Toolbar>
          <Typography variant="h6" color="inherit" noWrap>
            Handwritten digit recognition
          </Typography>
        </Toolbar>
      </AppBar>
      <main className={classes.layout}>
        <Grid container spacing={2}>
          <Grid item>
            <Paper className={classes.canvasPaper}>
              <CanvasDraw
                className={classes.canvasElement}
                ref={canvasRef}
                canvasHeight={canvansSize}
                canvasWidth={canvansSize}
                brushColor="#aaa"
                onChange={() => { recognize(model) }}
              />
              <canvas id="qwq" width={28} height={28} />
            </Paper>
          </Grid>
          <Grid item xs>
            <Paper className={classes.paper}>
              <ButtonGroup color="primary" className={classes.btnGroup}>
                <Button onClick={
                  () => {
                    canvasRef.current.undo()
                  }
                }>Undo</Button>
                <Button onClick={
                  () => {
                    canvasRef.current.clear()
                  }
                }>Clear</Button>
                <Button onClick={() => { recognize(model) }}>Recognize</Button>
              </ButtonGroup>
              {
                <Grid container spacing={2}>
                  {predRes.map(
                    (v, i) => (
                      <Grid item xs={6} key={i}>
                        <ProbIndicator v={v} i={i} />
                      </Grid>
                    )
                  )}
                </Grid>
              }
            </Paper>
          </Grid>
        </Grid>
      </main>
    </>
  );
}

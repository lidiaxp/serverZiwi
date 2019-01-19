alert(myVar1);

  var data = [ {"z":
  myVar1,

  colorscale: 'Jet',
    type: 'heatmap'
}
];

var layout = {
  plot_bgcolor:"black",
  paper_bgcolor:"#FFF3",
  title: {
    text:'Finalmente...',
    font: {
      family: 'Courier New, monospace',
      size: 24
    },
    xref: 'paper',
    x: 0.05,
  },
  xaxis: {
    title: {
      text: 'x Axis',
      font: {
        family: 'Courier New, monospace',
        size: 18,
        color: '#7f7f7f'
      }
    },
  },
  yaxis: {
    title: {
      text: 'y Axis',
      font: {
        family: 'Courier New, monospace',
        size: 18,
        color: '#7f7f7f'
      }
    }
  }
};

Plotly.newPlot('myDiv', data, layout);
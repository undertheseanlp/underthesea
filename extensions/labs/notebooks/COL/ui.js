var BrowserText = (function() {
      var canvas = document.createElement('canvas'),
      context = canvas.getContext('2d');

        /**
         * Measures the rendered width of arbitrary text given the font size and font face
         * @param {string} text The text to measure
         * @param {number} fontSize The font size in pixels
         * @param {string} fontFace The font face ("Arial", "Helvetica", etc.)
         * @returns {number} The width of the text
         **/
         function getWidth(text, fontSize, fontFace) {
          context.font = fontSize + 'px ' + fontFace;
          console.log(context.measureText(text));
          return context.measureText(text).width;
        }

      return {
        getWidth: getWidth
      };
    })();
    var svg = d3.select(element)
    .append("svg")
    .attr("width", 800)
    .attr("height", 300);

    var box = svg.append("rect")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("fill", "#424242");

    function displayText(rows) {
      var BOX_WIDTH = 600;
      var x = 10;
      var y = 50;
      var TEXT_HEIGHT = 14;
      var LINE_HEIGHT = 50;
      var SPACE_WIDTH = 10;
      for (var row of rows) {
        var token = row[0];
        var displayToken = token;
        if (token.length < 3) {
          displayToken = displayToken.padEnd(3, " ");
          displayToken = displayToken.padStart(6, " ");
        }
        var pos = row[1];
        var w = BrowserText.getWidth(displayToken, 16, "Time News Roman")
          // =================================================
          // DRAW TOKEN
          // =================================================
          var node = svg.append("text")
          .attr("x", x)
          .attr("y", y)
          .text(displayToken)
          .attr("style", "white-space:pre")
          .attr("fill", "#c4c4c4")
          .attr("font-family", "Time News Roman")
          .attr("font-size", 16)
          .attr("text", token)
          .on("mouseover", function(d, i) {
            d3.select(this)
            .attr("fill", "white")
            .attr("cursor", "pointer");
          })
          .on("mouseout", function(d, i) {
            d3.select(this)
            .attr("fill", "#c4c4c4");
          })
          .on("click", function(d, i) {
            var token = d3.select(this).attr("text");
            var url = 'https://vi.wikipedia.org/wiki/' + token;
            window.open(url, '_blank', 'toolbar=0,location=0,menubar=0,height=400,width=600');
          })
          // =================================================
          // DRAW TAG
          // =================================================
          var node_tag = svg.append("text")
          .attr("x", x + w / 2)
          .attr("y", y - 23)
          .text(pos)
          .attr("fill", "#c4c4c4")
          .attr("font-family", "Time News Roman")
          .attr("font-size", 12)

          function bracket(x1, y1, x2, y2) {
            var t1 = 2;
            var w = (y2 - y1 - 3 * t1) / 2;
            console.log(t1);
            var output = `
            M${x1} ${y1}
            l ${t1} -${t1}
            l ${w} 0
            l ${t1} -${t1}
            l ${t1} ${t1}
            l ${w} 0
            l ${t1} ${t1}
            `;
            return output;
          }
          var x1 = x;
          var y1 = y - TEXT_HEIGHT;
          var x2 = x;
          var y2 = y1 + w;
          console.log('y', y1);
          var path_text = bracket(x1, y1, x2, y2);
          var path = svg.append("path")
          .attr("d", path_text)
          .attr("stroke-width", "1")
          .attr("stroke", "#ccc")
          .attr("fill", "none");

          if (x + SPACE_WIDTH > BOX_WIDTH) {
            x = 10;
            y += LINE_HEIGHT;
          } else {
            x += w + SPACE_WIDTH;
          }
          console.log(0);
        }
      }
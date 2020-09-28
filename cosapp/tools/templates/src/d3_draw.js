var margin = 2;
var diameter = 400;

var svg = d3.select(".flexdiv").append("svg")
  .attr("width", diameter)
  .attr("height", diameter)

var svg = d3.select("svg"),
    g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

var pack = d3.pack()
    .size([diameter - margin, diameter - margin])
    .padding(10);

var calculateTextFontSize = function(d) {
  var id = d3.select(this).text();
  var radius = 0;
  if (d.fontsize){
    //if fontsize is already calculated use that.
    return d.fontsize;
  }
  if (!d.computed ) {
    //if computed not present get & store the getComputedTextLength() of the text field
    d.computed = this.getComputedTextLength();
    if(d.computed != 0){
      //if computed is not 0 then get the visual radius of DOM
      var r = d3.selectAll("#" + id).attr("r");
      //if radius present in DOM use that
      if (r) {
        radius = r;
      }
      //calculate the font size and store it in object for future
      d.fontsize = (2 * radius - 8) / d.computed * 16 + "px";
      return d.fontsize;
    }
  }
}

function DrawSystem(data) {

    var root = d3.hierarchy(data)
      .sum(function(d) { return d.size; })
      .sort(function(a, b) { return b.value - a.value; });

    var focus = root,
      nodes = pack(root).descendants(),
      view;

    var rainbow = d3.scaleSequential()
      .domain([-1, root.height])
      .interpolator(d3.interpolateRainbow);

    function get_opacity(d) {
        var c = d
        var opacity = 1

        while (c.parent && c.parent !== focus) {
            c = c.parent
            opacity = 0.9 * opacity
            }
        return opacity
        }

    function get_color(d) {
        var c = d
        while (c.parent && c.parent !== focus) {
            c = c.parent
        }

        if (c.parent) {
            var i = c.parent.children.indexOf(c)
            var rainbow = d3.scaleSequential()
              .domain([-1, c.parent.children.length])
              .interpolator(d3.interpolateRainbow);
            return c.parent ? rainbow(i) : "grey";
            }
        else {
            return "grey";
            }
        }

    function calculateTextFontSize(d, text) {
        if (d.fontsize){
            return d.fontsize;
            }
        d.fontsize = (2 * d.r * 0.8) / Math.max(text.getBBox().width, text.getBBox().height) * 16;
        return d.fontsize;
        }

  var circle = g.selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("class", function(d) { return d.parent ? d.children ? "node" : "node node--leaf" : "node node--root"; })
      .style("fill", "transparent")
      .style("stroke", function(d) { return get_color(d); })
      .style("stroke-opacity", function(d) { return get_opacity(d); })
      .style("stroke-width", "1.5px")
      .on("click", function(d) { if (focus !== d) zoom(d), d3.event.stopPropagation(); })
      ;

  var text = g.selectAll("text")
      .data(nodes)
      .enter().append("text")
      .attr("class", "label")
      .text(function(d){ return d.data.name; })
      .style("font-size", function(d){ return calculateTextFontSize(d, this) + "px"; })
      .style("fill", function(d) { return get_color(d); })
      .style("fill-opacity", function(d) { return (d === focus) || (d.parent === focus) ? 1 : 0; })
      .style("display", function(d) { return d.parent ? "inline" : "none"; })
      .attr("dy", ".35em")
      ;

  var node = g.selectAll("circle,text");

  circle.append("title")
      .text(function(d) { return d.data.full_name; });

  svg
      .style("background", "white")
      .on("click", function() { zoom(root); });

  zoomTo([root.x, root.y, root.r * 2 + margin]);

  function zoom(d) {
    var focus0 = focus; focus = d;

    var transition = d3.transition()
        .duration(d3.event.ctrlKey ? 2000 : 500)
        .tween("zoom", function(d) {
          var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);
          return function(t) { zoomTo(i(t)); };
        });

    transition.selectAll("circle")
        .style("stroke", function(d) { return get_color(d); })

    transition.selectAll("text")
        .style("fill", function(d) { return get_color(d); })
        .style("fill-opacity", function(d) { return ((d.parent === focus) || (d === focus && !(d.children))) ? 1 : 0; })
  }

  function zoomTo(v) {
    var k = diameter / v[2]; view = v;
    node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
    circle.attr("r", function(d) { return d.r * k; });
    text.style("font-size", function(d){ return calculateTextFontSize(d, this) * k  + "px"; })
    text.attr("dy", ".35em")
  }
};

DrawSystem(modelData);
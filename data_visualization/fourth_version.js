var width = 900,
    height = 105,
    cellSize = 12, // cell size
    week_days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'],
    month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    animDuration =1000,
	min = 0, 
	max=100;
	
var day = d3.time.format("%w"),
    week = d3.time.format("%U"),
    percent = d3.format(".1%"),
	format = d3.time.format("%Y%m%d");
	parseDate = d3.time.format("%Y%m%d").parse;
		
var color = d3.scale.sqrt().domain([1,length])
      .interpolate(d3.interpolateHcl)
      .range([d3.rgb("#007AFF"), d3.rgb('#FFF500')]);
	  
var svg = d3.select(".calender-map").selectAll("svg")
    .data(d3.range(2000, 2003))
  .enter().append("svg")
    .attr("width", '100%')
    .attr("data-height", '0.5678')
    .attr("viewBox",'0 0 900 105')
    .attr("class", "RdYlGn")
  .append("g")
    .attr("transform", "translate(" + ((width - cellSize * 53) / 2) + "," + (height - cellSize * 7 - 1) + ")");

svg.append("text")
    .attr("transform", "translate(-38," + cellSize * 3.5 + ")rotate(-90)")
    .style("text-anchor", "middle")
    .text(function(d) { return d; });
 
for (var i=0; i<7; i++)
{
svg.append("text")
    .attr("transform", "translate(-5," + cellSize*(i+1) + ")")
    .style("text-anchor", "end")
    .attr("dy", "-.25em")
    .text(function(d) { return week_days[i]; }); 
 }
 
d3.selection.prototype.moveToFront = function() {
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};

d3.selection.prototype.moveToBack = function() { 
    return this.each(function() { 
        var firstChild = this.parentNode.firstChild; 
        if (firstChild) { 
            this.parentNode.insertBefore(this, firstChild); 
        } 
    }); 
}; 
 
//http://stackoverflow.com/questions/10638529/how-to-parse-a-date-in-format-yyyymmdd-in-javascript
function dateParser(str) {
    var y = str.substr(0,4),
        m = str.substr(4,2) - 1,
        d = str.substr(6,2);
    var D = new Date(y,m,d);
    return (D.getFullYear() == y && D.getMonth() == m && D.getDate() == d) ? D : 'invalid date';
}
	
function prettyDate(dateObj){
	var newObj = new Date(dateObj);
	var day = newObj.getDate();
	var monthIndex = newObj.getMonth();
	var year = newObj.getFullYear();
	var monthNames = ["January", "February", "March","April", "May", "June", "July","August", "September", "October","November", "December"];
	return monthNames[monthIndex]+" "+day+", "+year;
}
 
function clickedDay(d){
	d3.selectAll(".day")
	   .transition()
	   .duration(animDuration/5)
	   .style("fill",function(d){
			var boxData = $("#box-id-"+dateParser(d).toDateString().replace(/ /g,"")).data();
			var sel = d3.select(this);
				sel.moveToBack();
			return boxData.fill;
		})
		.style("opacity",function(d){
			var boxData = $("#box-id-"+dateParser(d).toDateString().replace(/ /g,"")).data();
			return boxData.opacity;
		});

	var sel =d3.select(this);
	sel.moveToFront();

	d3.select("#box-id-"+dateParser(d).toDateString().replace(/ /g,""))
		.transition()
		.duration(animDuration/3)
		.style("fill","#82B446")
		.style("opacity","1");
		
	var format_decimal = d3.format(",.2f");

	var boxData = $("#box-id-"+dateParser(d).toDateString().replace(/ /g,"")).data();
	if(boxData.value !== undefined){
		$("#calendar-click-info").text("On " + prettyDate(boxData.date)+", " +  format_decimal((boxData.number_flights_cancelled / boxData.number_flights) * 100) + "% of the flights were cancelled");
	}
	else{
		$("#calendar-click-info").text("No information present for "+prettyDate(boxData.date));
	}
}

var rect = svg.selectAll(".day")
    .data(function(d) { return d3.time.days(new Date(d, 0, 1), new Date(d + 1, 0, 1)); })
  .enter()
	.append("rect")
    .attr("class", "day")
    .attr("width", cellSize)
    .attr("height", cellSize)
    .attr("x", function(d) { return week(d) * cellSize; })
    .attr("y", function(d) { return day(d) * cellSize; })
	.attr("id",function(d){ return "box-id-"+d.toDateString().replace(/ /g,""); })
    .attr("fill",'#fff')
	.attr("data-value", function(d){return d.sum;})
	.attr("data-date",function(d){return d;})
	.attr("data-fill","#fff")	
    .datum(format)
	.on("click", clickedDay);
	
var legend = svg.selectAll(".legend")
      .data(month)
    .enter().append("g")
      .attr("class", "legend")
      .attr("transform", function(d, i) { return "translate(" + (((i+1) * 50)+8) + ",0)"; });

legend.append("text")
   .attr("class", function(d,i){ return month[i] })
   .style("text-anchor", "end")
   .attr("dy", "-.25em")
   .text(function(d,i){ return month[i] });
   
svg.selectAll(".month")
    .data(function(d) { return d3.time.months(new Date(d, 0, 1), new Date(d + 1, 0, 1)); })
  .enter().append("path")
    .attr("class", "month")
    .attr("id", function(d,i){ return month[i] })
    .attr("d", monthPath);

d3.csv("format_data.csv", function(error, csv) {

  csv.forEach(function(d) {
    d.sum = parseInt(d.sum);
  });

  var Comparison_Type_Max = d3.max(csv, function(d) { return d.sum; });
 
  var data = d3.nest()
    .key(function(d) { 
		if(d.Month.toString().length == 1){
			d.Month = "0" + d.Month;
		}
		if(d.DayofMonth.toString().length == 1){
			d.DayofMonth = "0" + d.DayofMonth;
		}
		return d.year + d.Month + d.DayofMonth; })
    .rollup(function(d) { return  Math.sqrt(d[0].sum / Comparison_Type_Max); })
    .map(csv);

  var number_flights_cancelled = d3.nest()
    .key(function(d) { 
		if(d.Month.toString().length == 1){
			d.Month = "0" + d.Month;
		}
		if(d.DayofMonth.toString().length == 1){
			d.DayofMonth = "0" + d.DayofMonth;
		}
		return d.year + d.Month + d.DayofMonth; })
    .rollup(function(d) { return  d[0].sum; })
    .map(csv);
	
  var number_flights = d3.nest()
    .   key(function(d) {
		if(d.Month.toString().length == 1){
			d.Month = "0" + d.Month;
		}
		if(d.DayofMonth.toString().length == 1){
			d.DayofMonth = "0" + d.DayofMonth;
		}
		return d.year + d.Month + d.DayofMonth; })
    .rollup(function(d) { return  d[0].n; })
    .map(csv);
	
  rect.filter(function(d) { return d in data; })
      .attr("fill", function(d) { return color(data[d]); })
	  .attr("data-fill", function(d) { return color(data[d]); })
	  .attr("data-value", function(d) { return Math.round(data[d]*100)})
	  .attr("data-number_flights_cancelled", function(d) { return number_flights_cancelled[d]})
	  .attr("data-number_flights", function(d) { return number_flights[d]});   
});

function numberWithCommas(x) {
    x = x.toString();
    var pattern = /(-?\d+)(\d{3})/;
    while (pattern.test(x))
        x = x.replace(pattern, "$1,$2");
    return x;
}

function monthPath(t0) {
  var t1 = new Date(t0.getFullYear(), t0.getMonth() + 1, 0),
      d0 = +day(t0), w0 = +week(t0),
      d1 = +day(t1), w1 = +week(t1);
  return "M" + (w0 + 1) * cellSize + "," + d0 * cellSize
      + "H" + w0 * cellSize + "V" + 7 * cellSize
      + "H" + w1 * cellSize + "V" + (d1 + 1) * cellSize
      + "H" + (w1 + 1) * cellSize + "V" + 0
      + "H" + (w0 + 1) * cellSize + "Z";
}

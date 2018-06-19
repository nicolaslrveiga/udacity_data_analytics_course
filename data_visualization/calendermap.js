var width = 900,
    height = 105,
    cellSize = 12,
    week_days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'],
    month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    animDuration =1000,
	min = 0,
	max=100;

//Variables to parse Date objects.
var day = d3.time.format("%w"),
    week = d3.time.format("%U"),
    percent = d3.format(".1%"),
	format = d3.time.format("%Y%m%d");

//Maps input to a color between a certain range
var color = d3.scale.sqrt()
    .domain([0.03, 1])
    .interpolate(d3.interpolateHcl)
    .range([d3.rgb("#007AFF"), d3.rgb('#FFF500')]);
	  
//Append n svg elements with g element inside and translated by coordenates that depend on width, height and cellSize.
//n is the number of years.
var svg = d3.select(".calender-map").selectAll("svg")
    .data(d3.range(2000, 2003))
    .enter().append("svg")
    .attr("width", '100%')
    .attr("data-height", '0.5678')
    .attr("viewBox",'0 0 900 105')
    .attr("class", "RdYlGn")
    .append("g")
    .attr("transform", "translate(" + ((width - cellSize * 53) / 2) + "," + (height - cellSize * 7 - 1) + ")");

//appending text element to svg .calender-map, it is going to display the year for each calendar
svg.append("text")
    .attr("transform", "translate(-38," + cellSize * 3.5 + ")rotate(-90)")
    .style("text-anchor", "middle")
    .text(function(d) { return d; });

//appending text elements that are going to represent the day of the week.
for (var i = 0; i < 7; i++){
    svg.append("text")
        .attr("transform", "translate(-5," + cellSize*(i+1) + ")")
        .style("text-anchor", "end")
        .attr("dy", "-.25em")
        .text(function(d) { return week_days[i]; }); 
}

//bring SVG element to the front
d3.selection.prototype.moveToFront = function() {
    return this.each(function(){
        this.parentNode.appendChild(this);
	});
};

//move SVG elemento to back
d3.selection.prototype.moveToBack = function() { 
    return this.each(function() { 
        var firstChild = this.parentNode.firstChild; 
        if (firstChild) { 
            this.parentNode.insertBefore(this, firstChild); 
        } 
    }); 
}; 

//Parse a string in yyyymmdd format and returns a Date object
//http://stackoverflow.com/questions/10638529/how-to-parse-a-date-in-format-yyyymmdd-in-javascript
function dateParser(str) {
    var y = str.substr(0,4),
        m = str.substr(4,2) - 1,
        d = str.substr(6,2);
    var D = new Date(y,m,d);
    return (D.getFullYear() == y && D.getMonth() == m && D.getDate() == d) ? D : 'invalid date';
}

//Receives a data object and returns in a redable string.	
function prettyDate(dateObj){
    var newObj = new Date(dateObj);
    var day = newObj.getDate();
    var monthIndex = newObj.getMonth();
    var year = newObj.getFullYear();
    var monthNames = ["January", "February", "March","April", "May", "June", "July","August", "September", "October","November", "December"];
    return monthNames[monthIndex]+" "+day+", "+year;
}

//.day is going to be represented by a rect element
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

//append legend, name of each month 
var legend = svg.selectAll(".legend")
    .data(month)
    .enter().append("g")
    .attr("class", "legend")
    .attr("transform", function(d, i) { return "translate(" + (((i+1) * 50)+8) + ",0)"; });

legend.append("text")
    .attr("class", function(d,i){ return month[i]; })
    .style("text-anchor", "end")
    .attr("dy", "-.25em")
    .text(function(d,i){ return month[i]; });

//append path element, draw line between each month
svg.selectAll(".month")
    .data(function(d) { return d3.time.months(new Date(d, 0, 1), new Date(d + 1, 0, 1)); })
    .enter().append("path")
    .attr("class", "month")
    .attr("id", function(d,i){ return month[i]; })
    .attr("d", monthPath);

//loads format_data.csv
d3.csv("format_data.csv", function(error, csv) {
	//parsing variable sum as integer
    csv.forEach(function(d) {
        d.sum = parseInt(d.sum);
    });

    var Comparison_Type_Max = d3.max(csv, function(d) { return d.sum; });
	
	//ratio d.sum / max(d.sum), key is going to be the date
    var data = d3.nest()
        .key(function(d) {
            if(d.Month.toString().length == 1){
			    d.Month = "0" + d.Month;
		    }
		    if(d.DayofMonth.toString().length == 1){
			     d.DayofMonth = "0" + d.DayofMonth;
		    }
		    return d.year + d.Month + d.DayofMonth;
		})
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
            return d.year + d.Month + d.DayofMonth;
		})
        .rollup(function(d) { return  d[0].sum; })
        .map(csv);

    var number_flights = d3.nest()
        .key(function(d) {
            if(d.Month.toString().length == 1){
                d.Month = "0" + d.Month;
            }
            if(d.DayofMonth.toString().length == 1){
                d.DayofMonth = "0" + d.DayofMonth;
            }
            return d.year + d.Month + d.DayofMonth;
        })
        .rollup(function(d) { return  d[0].n; })
        .map(csv);

	//Format number of decimals
    var format_decimal = d3.format(",.2f");	
		
    rect.filter(function(d) { return d in data; })
        .attr("fill", function(d) { return color(data[d]); })
        .attr("data-fill", function(d) { return color(data[d]); })
        .attr("data-value", function(d) { return Math.round(data[d]*100); })
        .attr("data-number_flights_cancelled", function(d) { return number_flights_cancelled[d]; })
        .attr("data-number_flights", function(d) { return number_flights[d]; })
	    .attr("data-title", function(d) { return "On " + prettyDate(dateParser(d))+", " +  format_decimal((number_flights_cancelled[d] / number_flights[d]) * 100) + "% of the flights were cancelled"});
	$("rect").tooltip({container: 'body', html: true, placement:'top'});
		
	//Color legend
	var log = d3.scale.sqrt()
		.domain([0.00, 1])
		.interpolate(d3.interpolateHcl)
		.range([d3.rgb("#007AFF"), d3.rgb('#FFF500')]);

	var logLegend = d3.legend.color()
		.cells([0.00, 0.01, 0.02, 0.05, 0.1, 0.5, 1])
		.scale(log)
		.orient('horizontal')
		.shapePadding(5)
		.shapeWidth(50)
		.shapeHeight(20)
		.labelOffset(12);

	var new_legend = d3.select("#new_legend")
		.append("svg")
		.attr("width", '100%')
		.call(logLegend);		
});


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

Summary

The goal of this document is to explain the visualization created over the dataset which contains information on United State 
flight delays and performance, the dataset comes from RITA[1]. We did a data exploration fase using R(final_project.Rmd file 
and final_project.html) where we explore the dataset. The goal of this fase was to understand the dataset and define with 
message we wanted to show. The message that we wanted to show is the increase in the number of flights and cancellations over 
the years and also the effects of 9/11, in the end it became the effects  of 09/11 on the number of flights cancelled in the 
previous year(2000), 2001 and in the next year(2002).


Design

During the exploration fase, we found three messages that we wanted to show, they were: the number of flights increase between 1987 and 2008,
the number of flights cancelled increase and there was a peak in the number of flights cancelled in 2001. 

The number of flights and the number of flights cancelled is a data that is easilly observed when we sum up all the data points for a given
year. When comparing day by day, the increment is very slow and there is not much information that is worth visualize when comparing only two years
(This problem can be seen on first_version.html).
All things considered, the best type of chart to represent the increase of flights in a year and the increase in the number of cancellations 
is a line chart, where in the x axes it would be the years and the y axes the sommatory of all data points in that given year.

The problem with this type of chart is that it would be hard to see why there is a peak in the number of cancellations in 2001. Taking this in
consideration we decided to change the type of chart to one that enabled to observe both informations.

Doing a little reasearch we came across a calendar view headtmap[3]. With this type of chart we have the information day by day and
also year by year. It is possible to analyse 2001 and see what happened there and it also possible to see the difference in the number
of cancellation flights and total number of flights.

The first thing we want the reader to notice when looking at our visualization is that there is three distinct moments between 2000 and 2002. 
The first moment is a large region with a percentage of cancellation of flights that is higher that the third region, also the second and third
region are separeted by an event that reaches a very high number of cancellation flights. After noticing that, the reader is going to interact
with the visualization, he is going to realize that the event is 9/11 where in some point 99.99% of flights were cancelled. My final goal is that 
the reader is going to notice that after 9/11, there is a constant reduction in the number of cancellation compared with data before the event.

In the task of converting the visualization from a exploratory to a explanatory visualization, it was added a little introduction before the graph
to situate the reader and guide him to what we want to present.

Feedback

The first version of the visualization can be found in the file first_version.html.
There was three problems with this visualization, the first one was that there was a lot of data that does not give any information. The color
from 1987 until 1994 does not appears to change and the same problem happens for the rest of the years. The second feedback was more focused in
the color, it does not transmit well the type of information that I want to show. The third feedback was that it is hard to see a difference with
this scale of colors.

Looking at the feedbacks I observed that the biggest problem was regarding the amount of information in the screen. Having from 1987 until 2008
makes the visualization crowded and because the difference in the amount of cancellation flights is not that big from year to year, the color appears 
to be the same. In this point I decided to only show three year, 2000, 2001 and 2002. By doing that I decided to only focus in transmitting one
information, that is the effect of 9/11 on the number of flights cancelled. On my perception the color was fine, only showing three years it is
possible to observe the difference in color between 2000 and 2001.

The second version can be found in the file second_version.html.
The feedback this time was that reducing the number of charts improved the visualization a litle. The problem regarding color is still present,
it is hard to see a difference between 2000 and 2002. Another feedback was that the user was felling a need for a legend. The third feedback was
regarding 9/11, it is hard to drag the eyes over the visualization to see with day they are looking at and they also had to count the days to see
with day it was.

Taking the feedbacks in consideration I did three modification. I decided to change the color palete, this time I also used a square scale instead of
a linear scaled, it showed a significant improvement in the difference between 2000 and 2002. The second modification was instead of mouse over I changed
to a click event. I decided to do that because the amount of information that I need to show is a lot to a mouse over event. The third modification was
that I add a legend to the visualization.

My third version can be found in the file third_version.html
This time only one feedback was negative and was regarding the legend, the user sad that could not link the yellow from the legend with orange of the chart,
sad that if there was one more square in the legend with the orange color it would get better.

A fourth version was created based on feedback from the reviewer. The fourth version can be found on fourth_version.html. There were three 
feedbacks from the reviewer for this version. The first was regarding the legend, the legend should have numbers indicating the value the color
is representing. The second is related to mouse click event, it was pointed out  that the user associates more the mouse over event in this
type of charts. The third feedback is in relation to how the data is presented itself, the project is aiming in a explanatory visualization
and the current version is a exploratory visualization.

My final visualization can be found in the file final.html. 


Resources

jquery-1.10.2.js
bootstrap.css
bootstrap.min.js
https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js

References

[1] https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp
[2] https://www.amcharts.com/demos/line-chart-with-scroll-and-zoom/
[3] https://www.crowdanalytix.com/communityBlog/10-steps-to-create-calendar-view-heatmap-in-d3-js
function flashLabel(){
    var time = 400;
    for(var i=0; i<1; i++){
        $("#live-label").animate({"color": "#fff"}, time);
        $("#live-label").animate({"color": "#a2daed"}, time);
    }
    $("#live-label").animate({"color": "#fff"}, time);
}

$(document).ready(function(){
    flashLabel();
    $('[data-toggle="tooltip"]').tooltip();
});

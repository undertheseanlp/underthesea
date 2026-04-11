app.controller("SentimentCtrl", function ($scope) {
    var generateOutput = function(text){
        $scope.tag = "";
        $.ajax({
            type: "POST",
            url: "../sentiment",
            data: JSON.stringify({"text": text}),
            contentType: 'application/json'
        }).done(function (data) {
            try {
                console.log(data);
                var label = data["output"][0];
                var polarity = label.split("#")[1];
                $scope.tag = {
                    label: label,
                    polarity: polarity
                };
                $scope.$apply();
            } catch (e) {
                console.log(e);
            }
        }).fail(function () {
        });
    };
    $scope.text = "Sử dụng dịch rất tốt, thông tin minh bạch nhân viên nhiệt tình dễ thương nhiệt tình nói chung rất thích";
    $scope.do = function () {
        var text = $scope.text;
        generateOutput(text);
    };

    $scope.samples = [
        'Vay dc của bank không dễ đâu các chế ah.',
        'Không tin tưởng vào ngân hàng BIDV',
        'Vậy tốt quá, giờ sài thẻ an toàn lại tiện lợi nữa.'];

    $scope.updateText = function (text) {
        $scope.text = text;
    };
    $scope.init = function(){
        $scope.tag = "";
        $scope.do();
    };
    $scope.init();
});

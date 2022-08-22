app.controller("IPACtrl", function ($scope) {
    var generateOutput = function(text){
        $.ajax({
            type: "POST",
            url: "../ipa",
            data: JSON.stringify({"text": text}),
            contentType: 'application/json'
        }).done(function (data) {
            try {
            $scope.audio = document.getElementById('audio');
            $scope.audio.volume = 1;
            $scope.audio.src = "https://github.com/undertheseanlp/underthesea/releases/download/open-data-voice-ipa/success.mp3";
            $scope.audio.load();
            $scope.ipa = data['ipa'];
            $scope.found = true;
            $scope.notFound = false;
            $scope.$apply();
            } catch (e) {
                console.log(e);
                $scope.found = false;
                $scope.notFound = true;
            }

        }).fail(function () {
        });
    };

    $scope.init = function(){
        $scope.notFound = false;
        $scope.found = false;
        $scope.audio = "https://github.com/undertheseanlp/underthesea/releases/download/open-data-voice-ipa/horse.mp3";
    };
    $scope.init();
    $scope.do = function () {
        var text = $("#text").val();
        $scope.text = text;
        generateOutput(text);
    }

    $scope.playAudio = function(){
        console.log("play audio");
        $scope.audio = document.getElementById('audio');
        $scope.audio.play();
    }
});

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
            $scope.audio.src = data['audio_src']
            $scope.audio.load();
            $scope.audio.oncanplay = function(){
                $scope.showAudio = true;
                $scope.$apply();
            }
            $scope.ipa = data['ipa'];
            setTimeout(function(){
                $scope.found = true;
                $scope.isLoading = false;
                $scope.notFound = false;
                $scope.$apply();
            }, 1000);

            } catch (e) {
                console.log(e);
                $scope.found = false;
                $scope.isLoading = false;
                $scope.notFound = true;
            }

        }).fail(function () {
        });
    };

    $scope.init = function(){
        $scope.notFound = false;
        $scope.found = false;
        $scope.showAudio = false;
        $scope.isLoading = false;
        $scope.audio = "https://github.com/undertheseanlp/underthesea/releases/download/open-data-voice-ipa/horse.mp3";
    };
    $scope.init();
    $scope.do = function () {
        $scope.isLoading = true;
        var text = $("#text").val();
        $scope.text = text;
        generateOutput(text);
    }

    $scope.initDo = function(){
        $scope.init();
        $scope.do();
    }

    $scope.playAudio = function(){
        console.log("play audio");
        $scope.audio = document.getElementById('audio');
        $scope.audio.play();
    }
});

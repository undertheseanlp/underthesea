app.controller("IPACtrl", function ($scope) {
    var generateOutput = function(text){
        $.ajax({
            type: "POST",
            url: "../dictionary",
            data: JSON.stringify({"text": text}),
            contentType: 'application/json'
        }).done(function (data) {
            try {
            $scope.audio = document.getElementById('audio');
            $scope.audio.volume = 0.8;
            $scope.audio.src = "https://github.com/undertheseanlp/underthesea/releases/download/open-data-voice-ipa/success.mp3";
            $scope.audio.load();
            $scope.$apply();
            } catch (e) {
                console.log(e);
            }
        }).fail(function () {
        });
    };

    $scope.getPOSDescription = function(pos){
        var POS = {
            "N": "danh từ",
            "A": "tính từ",
            "V": "động từ",
            "P": "đại từ",
            "E": "giới từ",
            "I": "thán từ",
            "L": "định từ",
            "M": "số từ",
            "T": "trợ từ",
            "Z": "yếu tố cấu tạo từ",
            "R": "phó từ",
            "X": "từ không phân loại được",
            "C": "liên từ"
        };
        return POS[pos];
    };
    $scope.init = function(){
        $scope.notFound = false;
        $scope.found = false;
        $scope.audio = "https://github.com/undertheseanlp/underthesea/releases/download/open-data-voice-ipa/horse.mp3";
    };
    $scope.init();
    $scope.do = function () {
        var text = $("#text").val();
        generateOutput(text);
    }

    $scope.playAudio = function(){
        console.log("play audio");
        $scope.audio = document.getElementById('audio');
        $scope.audio.play();
    }
});

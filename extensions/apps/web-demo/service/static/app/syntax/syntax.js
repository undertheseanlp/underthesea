app.controller("WordSentCtrl", function ($scope) {
    var generateOutput = function (text) {
        $.ajax({
            type: "POST",
            url: "../chunking",
            data: JSON.stringify({"text": text}),
            contentType: 'application/json'
        }).done(function (data) {
            try {
                var tags = data["output"];
                var tokens = _.map(tags, function (tag) {
                    return tag[0]
                });
                var text = tokens.join(" ");


                var wordTags = _.map(tags, function (tag) {
                    return [tag[0], "Token"]
                });
                var wordEntities = generateEntitiesFromTags(wordTags);
                var wordSent = {
                    "config": WordSentBratConfig,
                    "doc": {
                        "text": text,
                        "entities": wordEntities
                    }
                };

                var posTags = tags;
                var posEntities = generateEntitiesFromTags(posTags);
                var pos = {
                    "config": POSTagBratConfig,
                    "doc": {
                        "text": text,
                        "entities": posEntities
                    }
                };

                var chunkingTags = _.map(tags, function (tag) {
                    return [tag[0], tag[2]];
                });
                var chunkEntities = generateEntitiesFromIOBTags(chunkingTags);
                var chunking = {
                    "config": ChunkingBratConfig,
                    "doc": {
                        "text": text,
                        "entities": chunkEntities
                    }
                };

                $("#word_sent_wrapper #word_sent").remove();
                $("#word_sent_wrapper").append("<div id='word_sent'></div>");
                Util.embed("word_sent", wordSent["config"], wordSent["doc"], []);
                $("#pos_tag_wrapper #pos_tag").remove();
                $("#pos_tag_wrapper").append("<div id='pos_tag'></div>");
                Util.embed("pos_tag", pos["config"], pos["doc"], []);
                $("#chunking_wrapper #chunking").remove();
                $("#chunking_wrapper").append("<div id='chunking'></div>");
                Util.embed("chunking", chunking["config"], chunking["doc"], []);
            } catch (e) {
                console.log(e);
            }
        }).fail(function () {
        });
    };

    $scope.samples = [
        'Phát hiện hai vật thể khả nghi tại nơi tàu ngầm Argentina mất tích',
        'Thống kê ngạc nhiên về Messi ở trận Siêu kinh điển',
        'Bảo Thanh từng “nước mắt như mưa” ôm con 4 tháng tuổi về đón Tết ở nhà ngoại'];

    $scope.text = "Nhật ký SEA Games ngày 21/8: Ánh Viên thắng giòn giã ở vòng loại.";
    $scope.updateText = function (text) {
        $scope.text = text;
    };

    generateOutput("Nhật ký SEA Games ngày 21/8: Ánh Viên thắng giòn giã ở vòng loại.");
    $scope.do = function () {
        var text = $scope.text;
        generateOutput(text);
    }
});

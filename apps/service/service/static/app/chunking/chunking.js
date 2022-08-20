app.controller("ChunkingCtrl", function ($scope) {
    var generateOutput = function (text) {
        $.ajax({
            type: "POST",
            url: "../chunking",
            data: JSON.stringify({"text": text}),
            contentType: 'application/json'
        }).done(function (data) {
            try {
                console.log(data);
                var tags = data["output"];
                var tokens = _.map(tags, function (tag) {
                    return tag[0]
                });
                var text = tokens.join(" ");
                var entities;

                entities = generateEntitiesFromTags(tags);
                var pos= {
                    "config": POSTagBratConfig,
                    "doc": {
                        "text": text,
                        "entities": entities
                    }
                };

                var chunkingTags = _.map(tags, function (tag) {
                    return [tag[0], tag[2]];
                });
                entities = generateEntitiesFromIOBTags(chunkingTags);
                var chunking = {
                    "config": ChunkingBratConfig,
                    "doc": {
                        "text": text,
                        "entities": entities
                    }
                };

                $("#chunking_wrapper #chunking").remove();
                $("#chunking_wrapper").append("<div id='chunking'></div>");
                Util.embed("chunking", chunking["config"], chunking["doc"], []);
                $("#pos_tag_wrapper #pos_tag").remove();
                $("#pos_tag_wrapper").append("<div id='pos_tag'></div>");
                Util.embed("pos_tag", pos["config"], pos["doc"], []);
            } catch (e) {
                console.log(e);
            }
        }).fail(function () {
        });

    };

    generateOutput("Nhật ký SEA Games ngày 21/8: Ánh Viên thắng giòn giã ở vòng loại.");

    $scope.do = function () {
        var text = $("#text").val();
        generateOutput(text);
    }
});

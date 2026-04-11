app.controller("NERTagCtrl", function ($scope) {
    var generateOutput = function (text) {
        $.ajax({
            type: "POST",
            url: "../ner",
            data: JSON.stringify({"text": text}),
            contentType: 'application/json'
        }).done(function (data) {
            try {
                window.start = 0;
                var tags = data["output"];
                var tokens = _.map(tags, function (tag) {
                    return tag[0]
                });
                var text = tokens.join(" ");

                var nerTags = _.map(tags, function (tag) {
                    return [tag[0], tag[3]];
                });
                var nerEntities = generateEntitiesFromIOBTags(nerTags);
                var ner = {
                    "config": NERTagBratConfig,
                    "doc": {
                        "text": text,
                        "entities": nerEntities
                    }
                };

                $("#ner_wrapper #ner").remove();
                $("#ner_wrapper").append("<div id='ner'></div>");
                Util.embed("ner", ner["config"], ner["doc"], []);
            } catch (e) {
                console.log(e);
            }
        }).fail(function () {
        });

    };

    $scope.samples = [
        "Tổng giám đốc đề xuất mua tàu cũ Trung Quốc về lại vị trí cũ",
        "Ủy ban Kiểm tra Trung ương công bố kết luận vụ ông Lê Phước Hoài Bảo",
        "Bộ Công Thương xóa một tổng cục, giảm nhiều đầu mối"
    ];
    $scope.text = "Nhật ký SEA Games ngày 21/8: Ánh Viên thắng giòn giã ở vòng loại.";
    $scope.updateText = function (text) {
        $scope.text = text;
    };
    generateOutput($scope.text);
    $scope.do = function () {
        generateOutput($scope.text);
    }
});

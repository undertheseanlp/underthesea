examples_text = """
Bức điện	N	3	nsubj
cũng	R	3	advmod
kêu gọi	V	0	root
Bộ	N	5	det:clf
Ngoại giao	N	3	obj
Mỹ	Np	5	compound
sớm	A	8	advmod:adj
triển khai	V	3	ccomp
đăng ký	V	8	acl:subj
và	C	11	cc
thu thập dữ liệu	N	9	conj
cá nhân	N	11	nmod
của	E	15	case
các	L	15	det
thông dịch viên	N	11	nmod:poss
Afghanistan	Np	15	compound
và	C	19	cc
những	L	19	det
người	Nc	11	conj
đủ	A	19	acl:subj
điều kiện	N	20	obj
xin	V	19	acl:subj
thị thực	V	22	obj
nhập cư	V	22	obj
đặc biệt	A	22	acomp
để	E	27	mark:pcomp
rời	V	22	advcl:objective
khỏi	V	27	compound:svc
quốc gia	N	27	obj
này	P	29	det:pmod
,	CH	33	punct
đồng thời	C	33	cc
cho	V	27	conj
biết	V	33	xcomp
Mỹ	Np	34	obj
nên	C	37	mark
khởi động	V	33	xcomp
các	L	39	det
chuyến	N	37	obj
bay	V	39	nmod
di tản	A	40	nmod
trong	E	43	case
thời gian	N	37	obl
muộn	A	43	amod
nhất	R	44	advmod
là	V	47	cop
ngày	N	33	ccomp
1/8	M	47	nmod
.	CH	3	punct
"""


def generate_svg_script(text):
    output = """
    var BrowserText = (function () {
    var canvas = document.createElement('canvas'),
        context = canvas.getContext('2d');

    /**
     * Measures the rendered width of arbitrary text given the font size and font face
     * @param {string} text The text to measure
     * @param {number} fontSize The font size in pixels
     * @param {string} fontFace The font face ("Arial", "Helvetica", etc.)
     * @returns {number} The width of the text
     **/
    function getWidth(text, fontSize, fontFace) {
        context.font = fontSize + 'px ' + fontFace;
        return context.measureText(text).width;
    }

    return {
        getWidth: getWidth
    };
    })();
    var svg = d3.select(element)
  .append("svg")
  .attr("width", 680)
  .attr("height", 200);

  var box = svg.append("rect")
  .attr("width", "100%")
  .attr("height", "100%")
  .attr("fill", "#424242");

    function displayText(tokens) {
        var text = tokens.join(" ");
        var BOX_WIDTH = 600;
        var x = 10;
        var y = 50;
        var space_width = 5;
        for (var token of tokens) {
            var node = svg.append("text")
                .attr("x", x)
                .attr("y", y)
                .text(token)
                .attr("fill", "#c4c4c4")
                .attr("font-family", "Time News Roman")
                .attr("font-size", 16)
            var w = BrowserText.getWidth(token, 16, "Time News Roman");
            if (x + space_width > BOX_WIDTH) {
                x = 10;
                y += 30;
            } else {
                x += w + space_width;
            }
            console.log(0);
        }
    }
    """
    output += "var text2 =`"
    output += text
    output += '`'
    output += """
    var tokens = $.map($.trim(text2).split("\\n"), function(val, i) {
        return $.trim(val).split("\\t")[0];
    });
    displayText(tokens);
    """
    return output

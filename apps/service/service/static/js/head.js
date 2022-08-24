/**
 Head JS        The only script in your <HEAD>
 Copyright    Tero Piirainen (tipiirai)
 License        MIT / http://bit.ly/mit-license
 Version        0.9

 http://headjs.com
 */(function (a) {
    var b = a.documentElement, c, d, e = [], f = [], g = {}, h = {},
        i = a.createElement("script").async === true || "MozAppearance" in a.documentElement.style || window.opera;
    var j = window.head_conf && head_conf.head || "head", k = window[j] = window[j] || function () {
            k.ready.apply(null, arguments)
        };
    var l = 0, m = 1, n = 2, o = 3;
    i ? k.js = function () {
        var a = arguments, b = a[a.length - 1], c = [];
        t(b) || (b = null), s(a, function (d, e) {
            d != b && (d = r(d), c.push(d), x(d, b && e == a.length - 2 ? function () {
                u(c) && p(b)
            } : null))
        });
        return k
    } : k.js = function () {
        var a = arguments, b = [].slice.call(a, 1), d = b[0];
        if (!c) {
            f.push(function () {
                k.js.apply(null, a)
            });
            return k
        }
        d ? (s(b, function (a) {
            t(a) || w(r(a))
        }), x(r(a[0]), t(d) ? d : function () {
            k.js.apply(null, b)
        })) : x(r(a[0]));
        return k
    }, k.ready = function (a, b) {
        if (a == "dom") {
            d ? p(b) : e.push(b);
            return k
        }
        t(a) && (b = a, a = "ALL");
        var c = h[a];
        if (c && c.state == o || a == "ALL" && u() && d) {
            p(b);
            return k
        }
        var f = g[a];
        f ? f.push(b) : f = g[a] = [b];
        return k
    }, k.ready("dom", function () {
        c && u() && s(g.ALL, function (a) {
            p(a)
        }), k.feature && k.feature("domloaded", true)
    });
    function p(a) {
        a._done || (a(), a._done = 1)
    }

    function q(a) {
        var b = a.split("/"), c = b[b.length - 1], d = c.indexOf("?");
        return d != -1 ? c.substring(0, d) : c
    }

    function r(a) {
        var b;
        if (typeof a == "object")for (var c in a)a[c] && (b = {name: c, url: a[c]}); else b = {name: q(a), url: a};
        var d = h[b.name];
        if (d && d.url === b.url)return d;
        h[b.name] = b;
        return b
    }

    function s(a, b) {
        if (a) {
            typeof a == "object" && (a = [].slice.call(a));
            for (var c = 0; c < a.length; c++)b.call(a, a[c], c)
        }
    }

    function t(a) {
        return Object.prototype.toString.call(a) == "[object Function]"
    }

    function u(a) {
        a = a || h;
        var b = false, c = 0;
        for (var d in a) {
            if (a[d].state != o)return false;
            b = true, c++
        }
        return b || c === 0
    }

    function v(a) {
        a.state = l, s(a.onpreload, function (a) {
            a.call()
        })
    }

    function w(a, b) {
        a.state || (a.state = m, a.onpreload = [], y({src: a.url, type: "cache"}, function () {
            v(a)
        }))
    }

    function x(a, b) {
        if (a.state == o && b)return b();
        if (a.state == n)return k.ready(a.name, b);
        if (a.state == m)return a.onpreload.push(function () {
            x(a, b)
        });
        a.state = n, y(a.url, function () {
            a.state = o, b && b(), s(g[a.name], function (a) {
                p(a)
            }), d && u() && s(g.ALL, function (a) {
                p(a)
            })
        })
    }

    function y(c, d) {
        var e = a.createElement("script");
        e.type = "text/" + (c.type || "javascript"), e.src = c.src || c, e.async = false, e.onreadystatechange = e.onload = function () {
            var a = e.readyState;
            !d.done && (!a || /loaded|complete/.test(a)) && (d(), d.done = true)
        }, b.appendChild(e)
    }

    setTimeout(function () {
        c = true, s(f, function (a) {
            a()
        })
    }, 0);
    function z() {
        d || (d = true, s(e, function (a) {
            p(a)
        }))
    }

    window.addEventListener ? (a.addEventListener("DOMContentLoaded", z, false), window.addEventListener("onload", z, false)) : window.attachEvent && (a.attachEvent("onreadystatechange", function () {
            a.readyState === "complete" && z()
        }), window.frameElement == null && b.doScroll && function () {
            try {
                b.doScroll("left"), z()
            } catch (a) {
                setTimeout(arguments.callee, 1);
                return
            }
        }(), window.attachEvent("onload", z)), !a.readyState && a.addEventListener && (a.readyState = "loading", a.addEventListener("DOMContentLoaded", handler = function () {
        a.removeEventListener("DOMContentLoaded", handler, false), a.readyState = "complete"
    }, false))
})(document)
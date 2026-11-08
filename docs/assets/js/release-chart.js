/*
 * Plot each model's average category score against its release date.
 *
 * Reads window.leaderboardData, the rows of _data/leaderboard.csv that the
 * page embeds with Liquid's jsonify filter. Every CSV value arrives as a
 * string. Point area is proportional to the number of backbone parameters,
 * and smaller models are drawn on top of larger ones so that they stay
 * visible where points overlap. A dashed line traces the state of the art
 * over time: each model that beats every model released before it.
 */
(function () {
  "use strict";

  var CATEGORIES = [
    ["pathology_classification", "Pathological (classification)"],
    ["anatomy_classification", "Non-pathological (classification)"],
    ["pathology_segmentation", "Pathological (segmentation)"],
    ["anatomy_segmentation", "Anatomical (segmentation)"],
  ];
  var MIN_RADIUS = 4;
  var MAX_RADIUS = 18;
  var ACCENT = "114, 83, 237"; // Just the Docs' default link color, #7253ed.

  function formatParams(params) {
    return (params / 1e6).toFixed(1) + "M";
  }

  function toModels(rows) {
    return rows.map(function (row) {
      var scores = CATEGORIES.map(function (category) {
        return {
          label: category[1],
          mean: parseFloat(row[category[0]]),
          se: parseFloat(row[category[0] + "_se"]),
        };
      });
      var total = scores.reduce(function (sum, score) {
        return sum + score.mean;
      }, 0);
      return {
        name: row.model,
        date: row.release_date,
        params: parseInt(row.params, 10),
        average: total / scores.length,
        scores: scores,
      };
    });
  }

  // The models that set a new best average score, in release order. Models
  // released on the same day are ordered best first, as in the notebook, so
  // only the best of them can join the frontier.
  function frontier(models) {
    var sorted = models.slice().sort(function (a, b) {
      return a.date === b.date ? b.average - a.average : a.date < b.date ? -1 : 1;
    });
    var points = [];
    sorted.forEach(function (model) {
      if (!points.length || model.average > points[points.length - 1].y) {
        points.push({ x: model.date, y: model.average });
      }
    });
    return points;
  }

  function render() {
    var canvas = document.getElementById("release-chart");
    if (!canvas || !window.Chart || !window.leaderboardData) return;

    var models = toModels(window.leaderboardData);
    var maxParams = Math.max.apply(null, models.map(function (m) { return m.params; }));
    var minParams = Math.min.apply(null, models.map(function (m) { return m.params; }));

    // Chart.js draws a dataset's points in order, so drawing the largest
    // models first leaves the smaller ones on top.
    var points = models
      .slice()
      .sort(function (a, b) { return b.params - a.params; })
      .map(function (model) {
        return {
          x: model.date,
          y: model.average,
          r: MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * Math.sqrt(model.params / maxParams),
          model: model,
        };
      });

    new Chart(canvas, {
      type: "bubble",
      data: {
        datasets: [
          {
            label: "Models",
            data: points,
            order: 1,
            backgroundColor: "rgba(" + ACCENT + ", 0.55)",
            borderColor: "rgb(" + ACCENT + ")",
            borderWidth: 1,
            hoverBackgroundColor: "rgba(" + ACCENT + ", 0.85)",
          },
          {
            type: "line",
            label: "State of the art",
            data: frontier(models),
            order: 2,
            borderColor: "rgba(110, 110, 110, 0.8)",
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0,
            pointHitRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: true },
        scales: {
          x: {
            type: "time",
            time: { unit: "month", displayFormats: { month: "MMM yyyy" } },
            ticks: { maxRotation: 0, autoSkipPadding: 24 },
            title: { display: true, text: "Release date" },
          },
          y: {
            title: { display: true, text: "Average category score" },
            grace: "5%",
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            displayColors: false,
            filter: function (item) {
              return item.datasetIndex === 0;
            },
            callbacks: {
              title: function (items) {
                return items[0].raw.model.name;
              },
              label: function (item) {
                var model = item.raw.model;
                return [
                  "Released " + model.date + ", " + formatParams(model.params) + " parameters",
                  "Average: " + model.average.toFixed(4),
                ].concat(
                  model.scores.map(function (score) {
                    return score.label + ": " + score.mean.toFixed(4) + " ± " + score.se.toFixed(4);
                  })
                );
              },
            },
          },
        },
      },
    });

    var caption = document.getElementById("release-chart-params");
    if (caption) {
      caption.textContent = formatParams(minParams) + " to " + formatParams(maxParams);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();

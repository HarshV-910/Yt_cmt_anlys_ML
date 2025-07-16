// document.addEventListener("DOMContentLoaded", async () => {
//   const outputDiv = document.getElementById("output");
//   const commentsListDiv = document.getElementById("comments-list");
//   const pieContainer = document.getElementById("piechart-container");
//   const pieChartCanvas = document.getElementById("piechart");
//   const API_KEY = "AIzaSyAgmJrE8oqan69HPqizGgjoxWL_oERDA-k"; // Your YouTube API key
//   const SENTIMENT_API_URL = "http://localhost:5000/predict";
//   let pieChart = null;

//   if (!outputDiv || !commentsListDiv || !pieContainer || !pieChartCanvas) {
//     alert("UI elements not found. Please check popup.html structure.");
//     return;
//   }

//   chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
//     const url = tabs[0].url;
//     const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
//     const match = url.match(youtubeRegex);

//     if (match && match[1]) {
//       const videoId = match[1];
//       outputDiv.innerHTML = `<span class="section-title">YouTube Video ID: <p style="display:inline">${videoId}</p></span><br><p>Fetching comments...</p>`;

//       const comments = await fetchComments(videoId);
//       if (!comments || comments.length === 0) {
//         outputDiv.innerHTML += "<p>No comments found for this video.</p>";
//         pieContainer.style.display = "none";
//         commentsListDiv.innerHTML = "";
//         return;
//       }

//       outputDiv.innerHTML += `<p>Found ${comments.length} comments. Analyzing...</p>`;

//       const predictions = await getSentimentPredictions(comments);

//       if (predictions && predictions.length === comments.length) {
//         const sentimentCounts = { 1: 0, 0: 0, 2: 0 };
//         predictions.forEach((s) => sentimentCounts[s]++);
//         const total = predictions.length;

//         const positivePercentage = ((sentimentCounts["1"] / total) * 100).toFixed(2);
//         const negativePercentage = ((sentimentCounts["0"] / total) * 100).toFixed(2);
//         const neutralPercentage = ((sentimentCounts["2"] / total) * 100).toFixed(2);

//         outputDiv.innerHTML = `
//           <p><b>Sentiment Analysis Results:</b></p>
//           <p>Positive: ${positivePercentage}%</p>
//           <p>Negative: ${negativePercentage}%</p>
//           <p>Neutral: ${neutralPercentage}%</p>
//         `;

//         // Show pie chart
//         pieContainer.style.display = "flex";
//         if (pieChart) pieChart.destroy();
//         pieChart = new Chart(pieChartCanvas, {
//           type: "pie",
//           data: {
//             labels: ["Positive", "Negative", "Neutral"],
//             datasets: [{
//               data: [sentimentCounts["1"], sentimentCounts["0"], sentimentCounts["2"]],
//               backgroundColor: [
//                 "rgba(67,233,123,0.85)",
//                 "rgba(255,88,88,0.85)",
//                 "rgba(189,189,189,0.85)"
//               ],
//               borderWidth: 2,
//               borderColor: "#fff"
//             }]
//           },
//           options: {
//             plugins: {
//               legend: {
//                 display: true,
//                 position: "bottom"
//               }
//             }
//           }
//         });

//         // Show top 10 comments
//         commentsListDiv.innerHTML = "";
//         comments.slice(0, 10).forEach((comment, idx) => {
//           const sentiment = predictions[idx];
//           let sentimentClass = "sentiment-neutral";
//           if (sentiment === 1) sentimentClass = "sentiment-positive";
//           else if (sentiment === 0) sentimentClass = "sentiment-negative";

//           const div = document.createElement("div");
//           div.className = `comment-item ${sentimentClass}`;
//           div.textContent = `${idx + 1}. ${comment}`;
//           commentsListDiv.appendChild(div);
//         });
//       } else {
//         outputDiv.innerHTML += "<p>Error: Prediction failed or mismatched.</p>";
//         pieContainer.style.display = "none";
//         commentsListDiv.innerHTML = "";
//       }
//     } else {
//       outputDiv.innerHTML = "<p>This is not a YouTube video page to use this extension.</p>";
//       pieContainer.style.display = "none";
//       commentsListDiv.innerHTML = "";
//     }
//   });

//   async function fetchComments(videoId) {
//     let comments = [];
//     let pageToken = "";
//     try {
//       while (comments.length < 2000) {
//         const response = await fetch(
//           `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`
//         );
//         const data = await response.json();
//         if (!data.items) break;
//         data.items.forEach((item) => {
//           comments.push(item.snippet.topLevelComment.snippet.textOriginal);
//         });
//         pageToken = data.nextPageToken;
//         if (!pageToken) break;
//       }
//     } catch (error) {
//       console.error("Error fetching comments:", error);
//     }
//     return comments;
//   }

//   async function getSentimentPredictions(comments) {
//     try {
//       const response = await fetch(SENTIMENT_API_URL, {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ comments }),
//       });
//       const result = await response.json();
//       if (Array.isArray(result) && typeof result[0] === "object" && "sentiment" in result[0]) {
//         return result.map((item) => item.sentiment);
//       }
//       return result;
//     } catch (error) {
//       console.error("Error fetching predictions:", error);
//       outputDiv.innerHTML += "<p>Error fetching sentiment predictions.</p>";
//       return [];
//     }
//   }
// });






document.addEventListener("DOMContentLoaded", async () => {
  const API_KEY = "AIzaSyAgmJrE8oqan69HPqizGgjoxWL_oERDA-k";
  const API_BASE = "http://localhost:5000";
  const videoIdDiv = document.getElementById("video-id");

  const metricTotal = document.querySelector("#metric-total .metric-value");
  const metricUsed = document.querySelector("#metric-used .metric-value");
  const metricUnique = document.querySelector("#metric-unique .metric-value");
  const metricAvg = document.querySelector("#metric-avgwords .metric-value");
  const metricRating = document.querySelector("#metric-rating .metric-value");
  const exportStatus = document.getElementById("export-status");
  const graphSections = document.getElementById("graph-sections");
  const activeSections = {};

  const createImg = (src) => {
    const img = document.createElement("img");
    img.src = src;
    img.className = "graph-img";
    return img;
  };
window.analysisStartTime = performance.now();

  function toggleSection(key, contentGenerator) {
    if (activeSections[key]) {
      graphSections.removeChild(activeSections[key]);
      delete activeSections[key];
    } else {
      const section = contentGenerator();
      activeSections[key] = section;
      graphSections.appendChild(section);
    }
  }

  const fetchComments = async (videoId) => {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 2000) {
        const response = await fetch(
          `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`
        );
        const data = await response.json();
        if (!data.items) break;
        data.items.forEach((item) => {
          comments.push({
            text: item.snippet.topLevelComment.snippet.textOriginal,
            author: item.snippet.topLevelComment.snippet.authorDisplayName,
            timestamp: item.snippet.topLevelComment.snippet.publishedAt,
          });
        });
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (e) {
      console.error("Error fetching comments:", e);
    }
    return comments;
  };

  const analyzeSentiment = async (comments) => {
    const texts = comments.map((c) => c.text);
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: texts })
    });
    return await response.json();
  };

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const match = url.match(/v=([\w-]{11})/);
    if (!match) return;
    const videoId = match[1];
    videoIdDiv.textContent = `Video ID: ${videoId}`;

    const rawComments = await fetchComments(videoId);
    const cleaned = rawComments.map((c) => c.text.trim()).filter((c) => c.length > 0);
    const uniqueUsers = new Set(rawComments.map((c) => c.author)).size;

    const sentimentData = await analyzeSentiment(rawComments);
    const sentiments = sentimentData.map((s) => s.sentiment);
    const avgWords = (
      cleaned.map((c) => c.split(" ").length).reduce((a, b) => a + b, 0) / cleaned.length
    ).toFixed(1);
    const rating = (
      ((sentiments.filter((s) => s === 1).length * 2 +
        sentiments.filter((s) => s === 2).length) * 10) /
      (2 * sentiments.length)
    ).toFixed(2);

    metricTotal.textContent = rawComments.length;
    metricUsed.textContent = cleaned.length;
    metricUnique.textContent = uniqueUsers;
    metricAvg.textContent = avgWords;
    metricRating.textContent = rating+"/10";

    document.getElementById("btn-piechart").addEventListener("click", async () => {
      toggleSection("pie", () => {
        const div = document.createElement("div");
        const sentimentCounts = {};
        sentiments.forEach(s => {
          sentimentCounts[s] = (sentimentCounts[s] || 0) + 1;
        });
        fetch(`${API_BASE}/generate_chart`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sentiment_counts: sentimentCounts })
        })
          .then(res => res.blob())
          .then(blob => div.appendChild(createImg(URL.createObjectURL(blob))));
        return div;
      });
    });

    document.getElementById("btn-trend").addEventListener("click", () => {
      toggleSection("trend", () => {
        const div = document.createElement("div");
        const sentiment_data = rawComments.map((c, i) => {
          return sentimentData[i] ? {
            timestamp: c.timestamp,
            sentiment: sentimentData[i].sentiment
          } : null;
        }).filter(Boolean);

        fetch(`${API_BASE}/generate_trend_graph`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sentiment_data })
        })
          .then(res => res.blob())
          .then(blob => div.appendChild(createImg(URL.createObjectURL(blob))));
        return div;
      });
    });

    document.getElementById("btn-wordcloud").addEventListener("click", () => {
      toggleSection("wordcloud", () => {
        const div = document.createElement("div");
        fetch(`${API_BASE}/generate_wordcloud`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ comments: cleaned })
        })
          .then(res => res.blob())
          .then(blob => div.appendChild(createImg(URL.createObjectURL(blob))));
        return div;
      });
    });

    document.getElementById("btn-top-comments").addEventListener("click", () => {
      toggleSection("top-comments", () => {
        const list = document.createElement("div");
        sentimentData.slice(0, 25).forEach((entry, i) => {
          const div = document.createElement("div");
          const sentiment = entry.sentiment;
          div.className = "comment-item " + (sentiment === 1 ? "comment-positive" : sentiment === 0 ? "comment-neutral" : "comment-negative");
          div.textContent = `${i + 1}. ${entry.comment}`;
          list.appendChild(div);
        });
        return list;
      });
    });

    document.getElementById("btn-export").addEventListener("click", async () => {
      const startTime = window.analysisStartTime || performance.timing.navigationStart;
      const endTime = performance.now();
      const totalTimeSeconds = ((endTime - startTime) / 1000).toFixed(2);

      const posCount = sentiments.filter((s) => s === 1).length;
      const neutralCount = sentiments.filter((s) => s === 0).length;
      const negCount = sentiments.filter((s) => s === 2).length;

      const textReport = `
    YouTube Comment Analysis Report

    Video ID: ${videoId}
    ----------------------------------------
    Total Comments Fetched     : ${rawComments.length}
    Total Comments Used        : ${cleaned.length}
    Unique Commenters          : ${uniqueUsers}
    Average Words per Comment  : ${avgWords}
    Sentiment Rating / 10      : ${rating}

    Total Positive Comments    : ${posCount}
    Total Neutral Comments     : ${neutralCount}
    Total Negative Comments    : ${negCount}

    Time Taken for Analysis    : ${totalTimeSeconds} seconds
    `.trim();

      const blob = new Blob([textReport], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `youtube_analysis_${videoId}.txt`;
      a.click();

      exportStatus.textContent = "Exported summary successfully.";
      exportStatus.style.display = "block";
    });
  });
});

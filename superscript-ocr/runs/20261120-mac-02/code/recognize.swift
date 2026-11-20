import Foundation
import Vision
import ImageIO

let number = try NSRegularExpression(pattern: #"[0-9０-９⁰¹²³⁴⁵⁶⁷⁸⁹]+(?:[,，\-–−][0-9０-９⁰¹²³⁴⁵⁶⁷⁸⁹]+)*"#)
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]
for path in CommandLine.arguments.dropFirst() {
    let started = ProcessInfo.processInfo.systemUptime
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil),
          let image = CGImageSourceCreateImageAtIndex(source,0,nil) else { throw CocoaError(.fileReadCorruptFile) }
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequestRevision3
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.automaticallyDetectsLanguage = true
    request.usesCPUOnly = true
    try VNImageRequestHandler(cgImage:image,orientation:.up).perform([request])
    var lines: [[String:Any]] = []
    for obs in request.results ?? [] {
        var candidates: [[String:Any]] = []
        for (rank,candidate) in obs.topCandidates(3).enumerated() {
            let text=candidate.string
            var matches: [[String:Any]] = []
            for m in number.matches(in:text,range:NSRange(text.startIndex...,in:text)) {
                guard let range=Range(m.range,in:text),let localized=try? candidate.boundingBox(for:range) else {continue}
                let b=localized.boundingBox
                matches.append(["text":String(text[range]),"x":b.midX,"y":1-b.midY,"box":[b.minX,1-b.maxY,b.width,b.height]])
            }
            candidates.append(["rank":rank+1,"text":text,"confidence":candidate.confidence,"matches":matches])
        }
        let b=obs.boundingBox
        lines.append(["box":[b.minX,1-b.maxY,b.width,b.height],"candidates":candidates])
    }
    let out: [String:Any] = ["path":path,"revision":request.revision,"cpuOnly":true,"minimumTextHeight":request.minimumTextHeight,"elapsedMS":(ProcessInfo.processInfo.systemUptime-started)*1000,"lines":lines]
    print(String(decoding:try JSONSerialization.data(withJSONObject:out,options:[.sortedKeys]),as:UTF8.self))
    fflush(stdout)
}

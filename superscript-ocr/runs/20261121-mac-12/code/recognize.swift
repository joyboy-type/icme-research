import Foundation
import Vision
import ImageIO

let settings: [(String,Int,VNRequestTextRecognitionLevel)] = [
    ("accurate_r3",3,.accurate),("fast_r3",3,.fast),("accurate_r2",2,.accurate)
]
for path in CommandLine.arguments.dropFirst() {
    for (name,revision,level) in settings {
        let started=ProcessInfo.processInfo.systemUptime
        var result:[String:Any] = ["path":path,"configuration":name,"requestedRevision":revision,
            "cpuOnlyRequested":true,"languageCorrection":false,"automaticLanguageDetection":false,
            "languages":["en-US"],"preprocessing":"original pixels; orientation.up; no resizing",
            "engine":"Apple Vision VNRecognizeTextRequest","minimumTextHeight":0]
        do {
            guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil),
                  let image=CGImageSourceCreateImageAtIndex(source,0,nil) else { throw CocoaError(.fileReadCorruptFile) }
            result["pixelWidth"]=image.width;result["pixelHeight"]=image.height
            let request=VNRecognizeTextRequest();request.revision=revision;request.recognitionLevel=level
            request.recognitionLanguages=["en-US"];request.usesLanguageCorrection=false
            request.automaticallyDetectsLanguage=false;request.minimumTextHeight=0;request.usesCPUOnly=true
            try VNImageRequestHandler(cgImage:image,orientation:.up).perform([request])
            var lines:[[String:Any]]=[]
            for observation in request.results ?? [] {
                let box=observation.boundingBox
                lines.append(["box":[box.minX,1-box.maxY,box.width,box.height],
                    "candidates":observation.topCandidates(3).map { ["text":$0.string,"confidence":$0.confidence] }])
            }
            result["lines"]=lines;result["actualRevision"]=request.revision
            result["status"]="completed";result["completionEvidence"]="perform returned without exception; results serialized"
        } catch {
            result["status"]="error";result["error"]=String(describing:error);result["lines"]=[]
        }
        result["elapsedMS"]=(ProcessInfo.processInfo.systemUptime-started)*1000
        print(String(decoding:try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]),as:UTF8.self))
        fflush(stdout)
    }
}

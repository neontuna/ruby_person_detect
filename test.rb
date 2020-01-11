require "onnxruntime"
require "mini_magick"
require "pry"
require 'streamio-ffmpeg'

person_objects = 0
s = Time.now

FileUtils.rm_rf(Dir.glob('./tmp/*'))

model = OnnxRuntime::Model.new("model.onnx")
movie = FFMPEG::Movie.new("person.mp4")
movie.screenshot("./tmp/screenshot_%d.jpg", { vframes: (movie.duration).to_i, frame_rate: movie.frame_rate/24 }, { validate: false })
pixels = []

Dir.each_child('./tmp') do |filename|
  print '.'
  img = MiniMagick::Image.open("./tmp/#{filename}")
  pixels << img.get_pixels
end

result = model.predict("image_tensor:0" => pixels)
result["num_detections:0"].each_with_index do |n, idx|
  n.to_i.times do |i|
    person_objects += 1 if result["detection_classes:0"][idx][i].to_i == 1
  end
end

e = Time.now
puts "\nRuntime: #{e-s}"
puts "Detected: #{person_objects}"